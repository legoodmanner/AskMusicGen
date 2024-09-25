import lightning as L
from lightning.pytorch.utilities import grad_norm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torchmetrics import Accuracy
import matplotlib.pyplot as plt

from models.gen_models import get_gen_model
from models.layers import *

class DiscrimProbeModule(L.LightningModule):
    def __init__(self, gen_model, config):
        super().__init__()
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.metric = Accuracy(task="multiclass", num_classes=config.model.peft.repr_head.n_classes)
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
            # Aggregate
            repr = torch.mean(repr, dim=-2) # TODO would try out different aggregation methods.
        else:  # inpt is precomputed feature (already aggregated)
            assert len(inps.shape) == 3, f'Features should be aggregated in advance, shape need to be [B, layer, dim], got {inps.shape}'
            extract_layer = self.config.model.gen_model.extract_layer
            repr = inps[:,extract_layer,:]
        logits = self.probe_mlp(repr)  # bsz, seq_len, n_class
        return logits
    
    def training_step(self, batch, batch_idx):
        inps, labels = batch
        logits = self(inps)
        train_loss = self.criterion(logits, labels)
        self.log("train_loss", train_loss, on_step = True, on_epoch = False, batch_size = self.config.data.batch_size, prog_bar = True)
        return train_loss

    def configure_optimizers(self):
        optimizer= optim.AdamW(self.probe_mlp.parameters(),
                          lr=self.config.training.learning_rate,
                          weight_decay=self.config.training.weight_decay)
        return optimizer

    def validation_step(self, batch, batch_idx):
        inps, labels = batch
        logits = self(inps)
        val_loss = self.criterion(logits, labels)
        self.metric(logits, labels)
        self.log("val_loss", val_loss, on_step = True, on_epoch = False, batch_size = self.config.data.batch_size, prog_bar = True)
        self.log("val_acc", self.metric, prog_bar=True)
        return val_loss

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.metric.compute())

\



