import os
import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchaudio.datasets import GTZAN
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle

def get_dataModule(config):
    dataConfig = config.data
    dataModule_dict = {
        'genre_classification': GTZANDataModule,
        'GTZAN_MusicGen300M': GTZANMusicGen300MFeatureModule
    }
    return dataModule_dict[dataConfig.name](**dataConfig)

class GTZANDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, sample_rate: int = 22050, num_samples: int = 650000, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_to_index = {}
        self.setup()

    def prepare_data(self):
        GTZAN(root=self.data_dir, download=True)

    def setup(self, stage=None):
        self.train_dataset = GTZAN(root=self.data_dir, download=False, subset='training', folder_in_archive='genres_original')
        self.val_dataset = GTZAN(root=self.data_dir, download=False, subset='validation', folder_in_archive='genres_original')
        self.test_dataset = GTZAN(root=self.data_dir, download=False, subset='testing', folder_in_archive='genres_original')

        # Create label mapping
        all_labels = set()
        for dataset in [self.train_dataset]:
            all_labels.update(label for _, _, label in dataset)
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(all_labels))}
        
        # Apply transforms to the datasets
    def pad_or_truncate(self, audio):
        if audio.size(1) > self.num_samples:
            return audio[:, :self.num_samples]
        return torch.nn.functional.pad(audio, (0, self.num_samples - audio.size(1)))

    def collate_fn(self, batch):
        waveforms, labels = [], [],
        for waveform, sr, label in batch:
            waveform = self.pad_or_truncate(waveform)
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            waveforms.append(waveform)
            labels.append(torch.tensor(self.label_to_index[label]))
        
        waveforms = torch.stack(waveforms)
        labels = torch.stack(labels)
        
        return waveforms, labels
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def teardown(self, stage=None):
        # Clean up any resources, if necessary
        pass
    
class FeatureDataset(Dataset):
    def __init__(self, root, subset):
        super().__init__()
        self.root = root
        self.subset = subset
        self.fl = [fn for fn in os.listdir(os.path.join(self.root, self.subset)) if fn.endswith('pkl')]
    
    def __getitem__(self, index):
        
        data = torch.load(os.path.join(self.root, self.subset,self.fl[index]), weights_only=True)
        repr, label = data['repr'], data['label']
        return repr, label
    
    def __len__(self):
        return len(self.fl)


class GTZANMusicGen300MFeatureModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_to_index = {}
        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = FeatureDataset(root=self.data_dir, subset='train')
        self.val_dataset = FeatureDataset(root=self.data_dir, subset='valid')
        # self.test_dataset = FeatureDataset(root=self.data_dir, download=False, subset='testing', folder_in_archive='genres_original')

        # Create label mapping
        all_labels = set()
        for dataset in [self.train_dataset]:
            all_labels.update(label for _, label in dataset)
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(all_labels))}
                

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def teardown(self, stage=None):
        # Clean up any resources, if necessary
        pass

if __name__ == '__main__':
    
    class TestConfig:
        name= 'genre_classification'
        hparams = {'data_dir': '/home/lego/Database/Data'}
            
    dm = get_dataModule(TestConfig)
    # dm = GTZANDataModule(data_dir='/home/lego/Database/Data')
    # dm.prepare_data()
    dm.setup()
    dls = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]
    print(dm.label_to_index)
    for dl in dls:
       for data in dl:
            print([d.shape for d in data])
    