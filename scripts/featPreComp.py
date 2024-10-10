import os
import torch
import numpy as np
from omegaconf import OmegaConf
from models.gen_models import get_gen_model
from torch.utils.data import DataLoader, Dataset
import lightning as L
class FeatureExtractor(L.LightningModule):
    # Output: if 30-second GTZAN, the shape would be [25, 1016, 1024] -> [layers, seq_len, dim]
    def __init__(self, config, output_dir, subset='train', mean_agg=True) -> None:
        super().__init__()
        self.config = config
        self.repr_extractor = get_gen_model(self.config)
        self.repr_extractor.layer = None
        self.output_dir = output_dir
        self.subset = subset # train, valid
        self.mean_agg = mean_agg
        

    def forward(self, wav):
        repr = self.repr_extractor(wav)
        assert isinstance(repr, tuple)
        repr = torch.stack(repr) #[layer, bs, seq_len, 1024]
        repr = repr.permute(1,0,2,3)  #[bs, layer, seq_len, 1024]
        if self.mean_agg:
            repr = repr.mean(-2) #[bs, layer, 1024]
        return repr
    
    def on_predict_start(self):
        os.makedirs(os.path.join(self.output_dir, self.subset), exist_ok=True)

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            wav, meta = batch
            # Extract hidden states as feature
            repr = self(wav)   #[bs, layer, (seq_len), 1024]
            if not self.mean_agg:
                bs, layer, seq_len, dim = repr.shape
                mini_crop = min(seq_len, meta['beat_f'].shape[-1])
                meta['beat_f'] =  meta['beat_f'][..., :mini_crop]
                repr = repr[...,:mini_crop,:]
            else:
                bs, layer, dim = repr.shape
            # Beat time to sample and builf binray 1D array
            for idx, r in enumerate(repr):
                torch.save(
                    {
                        'repr':r.detach().cpu(), 
                        'meta': {k:v[idx] for k, v in meta.items()}
                    },
                    os.path.join(self.output_dir, self.subset, f'{batch_idx*bs+idx}.pkl'),
                )
    

if __name__ == '__main__':
    from data import get_dataModule
    output_path = "../scratch/GTZAN/MusicGenSmall"
    modelConfig = OmegaConf.load('configs/gens/MusicGenSmall.yaml')
    dataConfig = OmegaConf.load('configs/tasks/GTZAN_genre.yaml')
    config = OmegaConf.merge(modelConfig, dataConfig)
    model = FeatureExtractor(config, output_path, subset='train', mean_agg=False)
    dl = get_dataModule(config)
    # Assuming you have a DataLoader for your dataset
    print('extracting....')
    trainer = L.Trainer(accelerator="gpu", devices=1)  # Use GPU if available
    trainer.predict(model, dataloaders=dl.train_dataloader())
    
    model.subset = 'valid'
    trainer.predict(model, dataloaders=dl.val_dataloader())

    model.subset = 'test'
    trainer.predict(model, dataloaders=dl.test_dataloader())

    print(f"Features extracted and saved to {output_path}")
                


    