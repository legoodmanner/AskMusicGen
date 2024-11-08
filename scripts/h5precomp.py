import h5py
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from models.gen_models import get_gen_model
from torch.utils.data import DataLoader, Dataset
import lightning as L


class FeatureExtractor(L.LightningModule):
    def __init__(self, config, output_dir, subset='train', mean_agg=True) -> None:
        super().__init__()
        self.config = config
        self.repr_extractor = get_gen_model(self.config)
        self.repr_extractor.layer = None
        self.output_dir = output_dir
        self.subset = subset # train, valid, test
        self.mean_agg = mean_agg
        self.h5_file = None
        self.dataset_size = 0  # Keeps track of the number of items in the dataset
        self.h5_filenames = {
            'train': os.path.join(self.output_dir, 'train.h5'),
            'valid': os.path.join(self.output_dir, 'valid.h5'),
            'test': os.path.join(self.output_dir, 'test.h5')
        }
    
    def on_predict_start(self):
        os.makedirs(self.output_dir, exist_ok=True)
        # Open HDF5 file for the current subset
        self.h5_file = h5py.File(self.h5_filenames[self.subset], 'w')
        self.dataset_size = 0  # Reset for each subset
        
    def on_predict_end(self):
        # Close the HDF5 file when done
        if self.h5_file:
            self.h5_file.close()

    def forward(self, wav):
        # print("Extracting features...")
        repr = self.repr_extractor(wav)
        # print("Finished extracting features")
        if isinstance(repr, tuple):
            repr = torch.stack(repr) #[layer, bs, seq_len, 1024]
        repr = repr.permute(1,0,2,3)  #[bs, layer, seq_len, 1024]
        if self.mean_agg:
            repr = repr.mean(-2) #[bs, layer, 1024]
        return repr
    
    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            # print(f"Extracting features for batch {batch_idx}")
            wav, meta = batch
            # print("forwarding...")
            repr = self(wav)  #[bs, layer, (seq_len), 1024]
            # print("finished")
            if not self.mean_agg:
                bs, layer, seq_len, dim = repr.shape
                if 'beat_f' in meta:
                    mini_crop = min(seq_len, meta['beat_f'].shape[-1])
                    meta['beat_f'] = meta['beat_f'][..., :mini_crop]
                else: mini_crop=seq_len
                repr = repr[...,:mini_crop,:]
            else:
                bs, layer, dim = repr.shape
            
            for idx, r in enumerate(repr):
                # Create a group for each sample
                group_name = f'{self.dataset_size}'
                group = self.h5_file.create_group(group_name)
                
                # Save the representation as a dataset in the group
                group.create_dataset('repr', data=r.detach().cpu().numpy())
                
                # Save the metadata in the same group
                meta_group = group.create_group('meta')
                for k, v in meta.items():
                    meta_group.create_dataset(k, data=v[idx].detach().cpu().numpy() if isinstance(v[idx], torch.Tensor) else v[idx])

                self.dataset_size += 1

if __name__ == '__main__':
    from data import get_dataModule
    # output_path = "/home/lego/Database/MTG/VampNet"
    modelConfig = OmegaConf.load('configs/gens/VampNet.yaml')
    # dataConfig = OmegaConf.load('configs/tasks/MTG_genre.yaml')

    # For pace part:
    output_path = "../scratch/GS/key/VampNet"
    # modelConfig = OmegaConf.load('configs/gens/MusicGenSmall.yaml')
    dataConfig = OmegaConf.load('configs/tasks/GS_key.yaml')
    # dataConfig.data.required_key = ['key', 'scaled_tempo']
    config = OmegaConf.merge(modelConfig, dataConfig)
    print("batch:" , config.data.batch_size)
    
    os.makedirs(output_path, exist_ok=True)
    print(f"Initializing feature extraction model")
    model = FeatureExtractor(config, output_path, subset='train', mean_agg=True)
    print("DataModule loading...")
    dl = get_dataModule(config)

    trainer = L.Trainer(accelerator="gpu", devices=1)

    # print('extracting train...')
    # trainer.predict(model, dataloaders=dl.train_dataloader())

    # model.subset = 'valid'
    # print('extracting valid...')
    # trainer.predict(model, dataloaders=dl.val_dataloader())

    model.subset = 'test'
    print('extracting test...')
    trainer.predict(model, dataloaders=dl.test_dataloader())

    print(f"Features extracted and saved to {output_path}")