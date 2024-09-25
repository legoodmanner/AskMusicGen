import os
import torch
import numpy as np
from omegaconf import OmegaConf
from models.gen_models import get_gen_model
from torch.utils.data import DataLoader, Dataset
import lightning as L

class FeatureExtractor(L.LightningModule):
    # Output: if 30-second GTZAN, the shape would be [25, 1016, 1024] -> [layers, seq_len, dim]
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
        repr = repr.permute(1,0,2,3)  #[bs, layer, seq_len, 1024]
        return repr
    
    def on_predict_start(self):
        os.makedirs(os.path.join(self.output_dir, self.mode), exist_ok=True)

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            wav, label = batch
            b = len(wav)
            repr = self(wav)   #[bs, layer, seq_len, 1024]
            for idx, (r, l) in enumerate(zip(repr, label)):
                torch.save(
                    {
                        'repr':r.detach().cpu().mean(-2), #Whether to compute the mean!!!!!!!!!!
                        'label': l
                    },
                    os.path.join(self.output_dir, self.mode, f'{batch_idx*b+idx}.pkl'),
                )
    

if __name__ == '__main__':
    from data import get_dataModule
    output_path = "cache/GTZAN/MusicGenMedium"
    modelConfig = OmegaConf.load('configs/gens/MusicGenMedium.yaml')
    # print('Extracing Feature from the model')
    dataConfig = OmegaConf.load('configs/tasks/GTZAN_genre.yaml')
    # print('For the task')
    model = FeatureExtractor(modelConfig, output_path, mode='train')
    dl = get_dataModule(dataConfig)
    # Assuming you have a DataLoader for your dataset

    print('extracting....')
    trainer = L.Trainer(accelerator="gpu", devices=1)  # Use GPU if available
    trainer.predict(model, dataloaders=dl.train_dataloader())
    
    model.mode = 'valid'
    trainer.predict(model, dataloaders=dl.val_dataloader())

    model.mode = 'test'
    trainer.predict(model, dataloaders=dl.test_dataloader())

    print(f"Features extracted and saved to {output_path}")
                


    