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

class ProbeModule(L.LightningModule):
    def __init__(self, gen_model, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore='gen_model')  
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
    
    def forward(self, wavs):
        if not self.config.model.peft.get('use_feature'): 
            with torch.no_grad():
                repr = self.repr_extractor(wavs) # bsz, seq_len, dim
            repr = torch.mean(repr, dim=-2) # TODO would try out different aggregation methods.
        else:
            extract_layer = self.config.model.gen_model.extract_layer
            repr = wavs[:,extract_layer,:]
        logits = self.probe_mlp(repr)  # bsz, seq_len, n_class
        return logits
    
    def training_step(self, batch, batch_idx):
        wavs, labels = batch
        logits = self(wavs)
        train_loss = self.criterion(logits, labels)
        self.log("train_loss", train_loss, on_step = True, on_epoch = False, batch_size = self.config.data.batch_size, prog_bar = True)
        return train_loss

    def configure_optimizers(self):
        optimizer= optim.AdamW(self.probe_mlp.parameters(),
                          lr=self.config.training.learning_rate,
                          weight_decay=self.config.training.weight_decay)
        return optimizer

    def validation_step(self, batch, batch_idx):
        wavs, labels = batch
        logits = self(wavs)
        val_loss = self.criterion(logits, labels)
        self.metric(logits, labels)
        self.log("val_loss", val_loss, on_step = True, on_epoch = False, batch_size = self.config.data.batch_size, prog_bar = True)
        self.log("val_acc", self.metric, prog_bar=True)
        return val_loss

    def on_validation_epoch_end(self):
        self.log("val_acc_epoch", self.metric.compute())

    # def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
    #     audio, text = batch
    #     text_latent = self.text_encoder.encode(text).squeeze()
        
    #     if self.config.data.token_spec.num_chunk is not None:
    #         # TODO: Find a better way to num_chunk
    #         audio = audio.permute(0,2,1,3)
    #         batch, n_codebook, orig_chunk, time = audio.shape
    #         audio = audio[:,:,:min(orig_chunk, self.config.data.token_spec.num_chunk)]
    #         audio = audio.permute(0,2,1,3)

    #     batch = (audio, text_latent)
    #     return batch
   
    # def on_train_epoch_end(self):
    #     self.log("mask_prob", self.mask_scheduler.get_current_mask_prob(), prog_bar = False)
    #     self.mask_scheduler.step()



