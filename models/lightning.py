import lightning as L
from lightning.pytorch.utilities import grad_norm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchmetrics # import Accuracy, F1Score
import matplotlib.pyplot as plt

from models.gen_models import get_gen_model
from models.layers import *
from models.eval import BeatF1MedianScore, BeatFMeasure

class DiscrimProbeModule(L.LightningModule):
    def __init__(self, gen_model, config):
        super().__init__()
        self.config = config
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        # Define metric function 
        self.metric = getattr(
            torchmetrics, 
            config.model.peft.metric)(task="multiclass", num_classes=config.model.peft.repr_head.n_classes)
        # Whether use pre-computed feature
        if self.config.model.peft.get('use_feature'):
            print('using pre-compute feature to train')
        # if self.config.model.peft.get('post_processor'):
            
        # Set pre-trained model as extractor (freeze)
        self.repr_extractor = gen_model
        # Set trainable MLP
        self.probe_mlp = MLP(
            input_size= config.model.gen_model.output_dim,
            hidden_sizes= config.model.peft.repr_head.hidden_sizes,
            output_size= config.model.peft.repr_head.n_classes,
            dropout=config.model.peft.repr_head.dropout
        )
        self.save_hyperparameters(ignore='gen_model')  
    
    def forward(self, inps):
        if not self.config.model.peft.get('use_feature'): # inps is wave
            with torch.no_grad():
                repr = self.repr_extractor(inps) # bsz, seq_len, dim, layer selection already happened
            # Aggregate
            repr = torch.mean(repr, dim=-2) # TODO would try out different aggregation methods.
        else:  # inpt is precomputed feature (already aggregated)
            # assert len(inps.shape) == 3, f'Features should be aggregated in advance, shape need to be [B, layer, dim], got {inps.shape}'
            extract_layer = self.config.model.gen_model.extract_layer
            repr = inps[:,extract_layer,:]
        logits = self.probe_mlp(repr)  # bsz, n_class
        return logits
    
    def training_step(self, batch, batch_idx):
        inps, meta = batch
        logits = self(inps)
        train_loss = self.criterion(logits, meta['label'])
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
        val_loss = self.criterion(logits, meta['label'])
        self.metric(logits, meta['label'])
        self.log("val_loss", val_loss, on_step = True, on_epoch = False, batch_size = self.config.data.batch_size, prog_bar = True)
        self.log("val_acc", self.metric, prog_bar=True)
        return val_loss

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.metric.compute())


class SequentialProbeModule(L.LightningModule):
    def __init__(self, gen_model, config):
        super().__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.2, weight=torch.tensor([0.1, 0.9]))
        self.metric = BeatFMeasure(label_freq=config.model.gen_model.fps)
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

    





