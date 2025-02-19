import h5py
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from models.gen_models import get_gen_model
from torch.utils.data import DataLoader, Dataset
import lightning as L


class FeatureExtractor(L.LightningModule):
    def __init__(self, config, output_dir, subset='train', agg_type='mean') -> None:
        super().__init__()
        self.config = config
        self.repr_extractor = get_gen_model(self.config)
        self.repr_extractor.layer = None
        self.output_dir = output_dir
        self.subset = subset # train, valid, test
        self.h5_file = None
        self.dataset_size = 0  # Keeps track of the number of items in the dataset
        self.h5_filenames = {
            'train': os.path.join(self.output_dir, f'train_{agg_type}.h5'),
            'valid': os.path.join(self.output_dir, f'valid_{agg_type}.h5'),
            'test': os.path.join(self.output_dir, f'test_{agg_type}.h5')
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
        # breakpoint()
        repr = self.repr_extractor(wav)
        # print("Finished extracting features")
        if isinstance(repr, tuple):
            repr = torch.stack(repr) #[layer, bs, seq_len, 1024] or [layer, bs, 1024]
        repr = repr.transpose(0, 1)  # [bs, layer, (seq_len), 1024]
        return repr
    
    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            # print(f"Extracting features for batch {batch_idx}")
            wav, meta = batch
            # print("forwarding...")
            repr = self(wav)  #[bs, layer, (seq_len), 1024]
            # print("finished")
            if repr.dim() == 3:
                bs, layer, dim = repr.shape
            else:
                bs, layer, seq_len, dim = repr.shape
            
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
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--agg_type', type=str, default='mean')
    args = parser.parse_args()

    """"For local 4080"""
    output_path = "/home/lego/Database/GS/key/MusicGenSmall_seq"

    """"For PACE"""
    # output_path = "../scratch/GS/key/MusicGenLarge"

    modelConfig = OmegaConf.load('configs/gens/MusicGenSmall.yaml')
    dataConfig = OmegaConf.load('configs/tasks/GS_key.yaml')
    dataConfig.data.required_key = ['key']
    
    # Whether aggregation or not
    modelConfig.model.gen_model.aggregation = False
    # modelConfig.model.gen_model.agg_type = args.agg_type
    modelConfig.model.gen_model.extract_layer = 3

    config = OmegaConf.merge(modelConfig, dataConfig)

    os.makedirs(output_path, exist_ok=True)
    print(f"Initializing feature extraction model")
    model = FeatureExtractor(config, output_path, subset='train', agg_type=args.agg_type)
    print("DataModule loading...")
    dl = get_dataModule(config)

    trainer = L.Trainer(accelerator="gpu", devices=1)

    if args.subset == 'train' or args.subset is None:
        print('extracting train...')
        trainer.predict(model, dataloaders=dl.train_dataloader())

    # model.subset = 'valid'
    # print('extracting valid...')
    # trainer.predict(model, dataloaders=dl.val_dataloader())
    if args.subset == 'test' or args.subset is None:
        model.subset = 'test'
        print('extracting test...')
        trainer.predict(model, dataloaders=dl.test_dataloader())

    print(f"Features extracted and saved to {output_path}")