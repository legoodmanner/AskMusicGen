import lightning as L
from lightning.pytorch.utilities import grad_norm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from models.gen_models import get_gen_model
from models.layers import *
from models.eval import get_metric_from_task
from models.loss import MaskedMSEWithLogitsLoss

def get_loss_from_task(config):
    task2loss = {
        "GTZAN_rhythm": nn.CrossEntropyLoss,
        "GS_key": nn.CrossEntropyLoss,
        "GTZAN_genre": nn.CrossEntropyLoss,
        "MTG_genre": nn.BCEWithLogitsLoss,
        "GS_tempo": nn.CrossEntropyLoss,
    }

    loss_config = config.model.peft.get('loss')
    task = config.experiment.task.replace('_feature', '')
    loss = task2loss[task](**(loss_config if loss_config is not None else {}))
    return loss


class DiscrimProbeModule(L.LightningModule):
    def __init__(self, gen_model, config):
        super().__init__()
        self.config = config
        # Define loss function
        self.criterion = get_loss_from_task(config)
        # Define metric function 
        self.metric = get_metric_from_task(config)
        print('Using metric:',self.metric.__class__.__name__)
        # Whether use pre-computed feature
        if self.config.model.peft.get('use_feature'):
            print('using pre-compute feature to train') 
        # if self.config.model.peft.get('post_processor'):
            

        if not config.model.peft.get('use_feature'): # inps is wave
            self.repr_extractor = gen_model
        # Set trainable MLP
        self.probe_mlp = MLP(
            input_size= config.model.gen_model.output_dim,
            hidden_sizes= config.model.peft.repr_head.hidden_sizes,
            output_size= config.model.peft.repr_head.n_classes,
            dropout=config.model.peft.repr_head.dropout,
        )
        self.save_hyperparameters(config, ignore='gen_model')
        self.log('layer', self.config.model.gen_model.extract_layer)
        
    
    def forward(self, inps):
        if not self.config.model.peft.get('use_feature'): # inps is wave
            with torch.no_grad():
                repr = self.repr_extractor(inps) # bsz, seq_len, dim, layer selection already happened
            # Aggregate:
                # After this commit, the aggregation is done in the representation extractor
        else:  # inpt is precomputed feature (already aggregated)
            repr = inps
        logits = self.probe_mlp(repr)  # bsz, n_class
        return logits
    
    def training_step(self, batch, batch_idx):
        inps, meta = batch
        if 'label' in meta:
            label = meta['label']
        else:
            label = meta[self.config.data.required_key[0]]
        if 'GS_tempo' in self.config.experiment.task:
            # quantize to multiple of 5
            label = (label/5).round().long()
        logits = self(inps)
        train_loss = self.criterion(logits, label)
        self.log("train_loss", train_loss, on_step = False, on_epoch = True, batch_size = self.config.data.batch_size, prog_bar = True)
        return train_loss
    

    def configure_optimizers(self):
        optimizer= optim.AdamW(self.probe_mlp.parameters(),
                          lr=self.config.training.learning_rate,
                          weight_decay=self.config.training.weight_decay)
        return optimizer

    def validation_step(self, batch, batch_idx):
        inps, meta = batch
        if 'label' in meta:
            label = meta['label']
        else:
            label = meta[self.config.data.required_key[0]]
        logits = self(inps)
        if 'GS_tempo' in self.config.experiment.task:
            # quantize to multiple of 5
            tmp = torch.zeros(logits.shape[0], logits.shape[1]*5, device=logits.device)
            tmp[:, ::5] = logits
            logits = tmp
            val_loss = self.criterion(logits, label)
            # 5 times len
            

        else:
            val_loss = self.criterion(logits, label)

        self.metric(logits, label.long())
        self.log("val_loss", val_loss, on_step = False, on_epoch = True, batch_size = self.config.data.batch_size, prog_bar = True)
        
        return val_loss

    def on_validation_epoch_end(self):
        if self.metric.__class__.__name__ == 'MulticlassAccuracy': 
            self.log("val_acc", self.metric.compute(), prog_bar=True)
        elif self.metric.__class__.__name__ == 'MultilabelAUROC':
            auc = self.metric.compute()
            self.log("val_auc", auc, prog_bar=True)
        elif self.metric.__class__.__name__ == 'KeyAccRefined':
            key_acc = self.metric.compute()
            self.log("val_key_acc_refined", key_acc, prog_bar=True)
        elif self.metric.__class__.__name__ == 'TempoAcc':
            tempo_acc = self.metric.compute()
            self.log("val_tempo_acc", tempo_acc, prog_bar=True)



class SequentialProbeModule(L.LightningModule):
    def __init__(self, gen_model, config):
        super().__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.2, weight=torch.tensor([0.1, 0.9]))
        self.metric = get_metric_from_task(config)
        if self.config.model.peft.get('use_feature'):
            print('using pre-compute feature to train')
        self.repr_extractor = gen_model
        self.probe_mlp = MLP(
            input_size=self.repr_extractor.model.decoder.config.hidden_size,
            hidden_sizes= config.model.peft.repr_head.hidden_sizes,
            output_size= config.model.peft.repr_head.n_classes,
            dropout=config.model.peft.repr_head.dropout
        )
        self.save_hyperparameters(ignore='gen_model') 

    def forward(self, inps):
        if not self.config.model.peft.get('use_feature'): # inps is wave
            with torch.no_grad():
                repr = self.repr_extractor(inps) # bsz, seq_len, dim
        else:  # inpt is precomputed feature (already aggregated)
            # assert len(inps.shape) == 4, f'Features time dimension should not be squeezed, shape need to be [B, layer, sequencial, dim], got {inps.shape}'
            # extract_layer = self.config.model.gen_model.extract_layer
            repr = inps
        logits = self.probe_mlp(repr)  # bsz, seq, n_class
        return logits.transpose(1,2) # bsz, n_class, seq
    
    def training_step(self, batch, batch_idx):
        inps, meta = batch
        logits = self(inps)
        train_loss = self.criterion(logits, meta['beat_f'].long())
        self.log("train_loss", train_loss, on_step = True, on_epoch = False, batch_size = self.config.data.batch_size, prog_bar = True)
        return train_loss

    def configure_optimizers(self):
        optimizer= optim.AdamW(self.probe_mlp.parameters(),
                          lr=self.config.training.learning_rate,
                          weight_decay=self.config.training.weight_decay)
        return optimizer

    def validation_step(self, batch, batch_idx):
        inps, meta = batch
        logits = self(inps)
        val_loss = self.criterion(logits, meta['beat_f'].long())
        self.metric(logits, meta['beat_f'])
        self.log("val_loss", val_loss, on_step = True, on_epoch = False, batch_size = self.config.data.batch_size, prog_bar = True)
        return val_loss

    def on_validation_epoch_end(self):
        self.log("val_f1_epoch", self.metric.compute())

    





