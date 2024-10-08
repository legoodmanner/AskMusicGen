import os
import torch
import torchaudio
import lightning as L
from torch.utils.data import random_split, DataLoader
from utils.gtzan import GTZAN
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import h5py

def time_to_frame(seq_in_sec, fps, n_frame):
    """
    seq_in_sec: [number of beat events]

    """
    frame_idx = (torch.tensor(seq_in_sec) * fps).int()
    beat_f = torch.zeros(n_frame or fps*35)
    beat_f[frame_idx] = 1
    return beat_f
    
def get_dataModule(config):
    dataConfig = config.data
    dataModule_dict = {
        'genre_classification': GTZANDataModule,
        'genre_classification_on_feature': PreComputeDataModule,
        'beat_tracking_on_feature': PreComputeDataModule,
    }
    return dataModule_dict[dataConfig.name](config=config, **dataConfig)


def get_aggregators(methods=None):
    # get aggregators with 'list of str' or 'str' or None
    agg_dict = {
            'layer_mean': lambda a: torch.mean(a, axis=-3, keepdim=True),
            'time_mean': lambda a: torch.mean(a, axis=-2, keepdim=True),
    }
    aggregators = []
    if not methods: return aggregators
    if isinstance(methods, list):
        for method in methods:
            aggregators.append(agg_dict[method])
        aggregators.append(torch.squeeze)
    elif isinstance(methods, str):
        aggregators.append(agg_dict[methods])
        aggregators.append(torch.squeeze)
    else:
        raise TypeError('incompatiable type for aggregation methods')
    return aggregators

class GTZANDataModule(L.LightningDataModule):
    def __init__(self, config, data_dir: str, batch_size: int = 32, num_workers: int = 4, sample_rate: int = None, num_samples=650000, **kwargs):
        super().__init__()
        assert sample_rate is not None, 'Desired sample rate must be assigned manually'
        self.config = config
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
            all_labels.update(meta['label'] for _, meta in dataset)
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(all_labels))}
        
        # Apply transforms to the datasets
    def pad_or_truncate(self, audio):
        if audio.size(1) > self.num_samples:
            return audio[:, :self.num_samples]
        return torch.nn.functional.pad(audio, (0, self.num_samples - audio.size(1)))

    def collate_fn(self, batch):
        waveforms, meta = [], dict(label=[], beat_t=[], beat_f=[]),
        for waveform, info in batch:
            waveform = self.pad_or_truncate(waveform)
            waveform = torchaudio.functional.resample(waveform, info['sample_rate'], self.sample_rate)
            waveforms.append(waveform)
            meta['label'].append(torch.tensor(self.label_to_index[info['label']]))
            meta['beat_t'].append(info['beat_t'])
            meta['beat_f'].append(time_to_frame(info['beat_t'], fps=self.config.model.gen_model.fps, n_frame=None).clone().detach())
        # Wrap the collected data
        waveforms = torch.stack(waveforms)
        for k, v in meta.items():
            try:
                meta[k] = torch.stack(v)
            except:
                pass
        return waveforms, meta
        

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
        repr, meta = data['repr'], data['meta'] # repr.shape = [layers, seq_len, dim], label.shape = [1]
    
        return repr, meta #TODO not hard setting wished
   
    def __len__(self):
        return len(self.fl)
    
class FeatureHDF5Dataset(Dataset):
    def __init__(self, root, subset, extract_layer, required_key, transform=None):
        
        self.hdf5_path = os.path.join(root, subset + '.h5')
        assert os.path.isfile(self.hdf5_path)
        self.transform = transform
        self.extract_layer = extract_layer
        self.required_key = required_key
        with h5py.File(self.hdf5_path, 'r') as f:
            # filename as key (?)
            self.indexs = list(f.keys())

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            # group = f[self.indexs[idx]]
            
            # Load repr data
            repr_data = torch.from_numpy(f[self.indexs[idx]]['repr'][self.extract_layer])
            # Load meta data
            meta_data = {}
            for key in self.required_key:
                if key in ['beat_t', 'label']:
                    meta_data[key] = f[self.indexs[idx]]['meta'][key][:]
                else:
                    meta_data[key] = torch.tensor(f[self.indexs[idx]]['meta'][key])

            # for key, value in group['meta'].items():
            #     # print(key, value, value.__class__, key.__class__)
            #     if key in ['beat_t']:
            #         meta_data[key] = value
            #     elif key in ['label']:
            #         meta_data[key] = value
            #     else:
            #         meta_data[key] = torch.tensor(value)
                    

        if self.transform:
            repr_data = self.transform(repr_data)

        return repr_data, meta_data
    


class PreComputeDataModule(L.LightningDataModule):
    def __init__(self, config, data_dir: str, batch_size: int = 32, num_workers: int = 4, required_key=None, **kwargs):
        super().__init__()
        # data_dir should contain train / valid / test 3 directories 
        self.config = config
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_to_index = {}
        self.aggregators = get_aggregators(kwargs.get('aggregators')) # list or str or None
        self.required_key = required_key
        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pretrained_name = self.config.model.gen_model.name
        extract_layer = self.config.model.gen_model.extract_layer
        if 'raw' in os.listdir(self.data_dir) or 'train' not in os.listdir(self.data_dir):
            # still in the parent directory, should move to precompute feature folder
            self.data_dir = os.path.join(self.data_dir, pretrained_name)
       
        self.train_dataset = FeatureHDF5Dataset(root=self.data_dir, subset='train', extract_layer=extract_layer, required_key=self.required_key)
        self.val_dataset = FeatureHDF5Dataset(root=self.data_dir, subset='valid', extract_layer=extract_layer, required_key=self.required_key)
        self.test_dataset = FeatureHDF5Dataset(root=self.data_dir, subset='test', extract_layer=extract_layer, required_key=self.required_key)
        # Create label mapping
        # all_labels = set()
        # for dataset in [self.train_dataset]:
        #     all_labels.update(meta['label'] for _, meta in dataset)
            
        # self.label_to_index = {label: idx for idx, label in enumerate(sorted(all_labels))}
                

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def teardown(self, stage=None):
        # Clean up any resources, if necessary
        pass

    def collate_fn(self, batch):
       # input should be in the shaoe of [.., layers, seq_len, dim]
        for idx, (repr, meta) in enumerate(batch):
            if idx==0:
                reprs, metas = [], {k:[] for k, _ in meta.items()}
            reprs += [repr]
            for k, v in meta.items():
                metas[k] += [v]
        
        for k, v in metas.items():
            if k != 'beat_t':
                metas[k] = torch.stack(v)
        return torch.stack(reprs), metas
            

    



if __name__ == '__main__':
    from omegaconf import OmegaConf
    testconf = {
        'data': {
            'name': 'genre_classification',
            'data_dir': '../scratch/GTZAN/raw',
            'batch_size': 4,
            'num_workers': 0,
            'sample_rate': 32000,
        }
    }
    testconf = OmegaConf.create(testconf)
    dm = get_dataModule(testconf)
    # dm = GTZANDataModule(data_dir='/home/lego/Database/Data')
    # dm.prepare_data()
    dm.setup() 
    dls = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]
    print(dm.label_to_index)
    for dl in dls:
       for data in dl:
            print([d.shape for d in data])