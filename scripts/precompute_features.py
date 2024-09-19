import os
import torch
import numpy as np
from omegaconf import OmegaConf
from models.gen_models import get_gen_model
from torch.utils.data import DataLoader, Dataset
import lightning as L

class FeatureExtractor(L.LightningModule):
    def __init__(self, config, output_dir, mode='train') -> None:
        super().__init__()
        self.config = config
        self.repr_extractor = get_gen_model(self.config)
        self.repr_extractor.layer = None
        self.output_dir = output_dir
        self.mode = mode # train, valid
        

    def forward(self, wav):
        repr = self.repr_extractor(wav)
        assert isinstance(repr, tuple)
        repr = torch.stack(repr) #[layer, bs, seq_len, 1024]
        repr = repr.permute(1,0,2,3).mean(2)  #[bs, layer, 1024]
        return repr
    
    def on_predict_start(self):
        os.makedirs(os.path.join(self.output_dir, self.mode), exist_ok=True)

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            wav, label = batch
            b = len(wav)
            repr = self(wav)  # tuple( torch.Size([bs, seq_len, 1024]) * layer_number )
            for idx, (r, l) in enumerate(zip(repr, label)):
                torch.save(
                    {
                        'repr':r.detach().cpu(),
                        'label': l
                    },
                    os.path.join(self.output_dir, self.mode, f'{batch_idx*b+idx}.pkl'),
                )
    

if __name__ == '__main__':
    from data import get_dataModule
    output_path = "cache/GTZAN_MusicGen_s/"
    modelConfig = OmegaConf.load('/home/lego/Gatech/Rupak/configs/gens/musGen.yaml')
    dataConfig = OmegaConf.load('/home/lego/Gatech/Rupak/configs/tasks/GTZAN_genre.yaml')
    model = FeatureExtractor(modelConfig, output_path, mode='train')
    dl = get_dataModule(dataConfig)
    # Assuming you have a DataLoader for your dataset

    trainer = L.Trainer(accelerator="gpu", devices=1)  # Use GPU if available
    trainer.predict(model, dataloaders=dl.train_dataloader())
    
    model.mode = 'valid'
    trainer.predict(model, dataloaders=dl.val_dataloader())

    print(f"Features extracted and saved to {output_path}")
                


    