import os
import torch
import torchaudio
import lightning as L
from torch.utils.data import random_split, DataLoader
from utils.gtzan import GTZAN
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import h5py
import omegaconf


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
        'genre_classification_GTZAN': GTZANDataModule,
        'genre_classification_on_feature': PreComputeDataModule,
        'beat_tracking_on_feature': PreComputeDataModule,
        'genre_classification_MTG': MTGDataModule,
        'key_GS': GSDataModuel,
    }
    return dataModule_dict[dataConfig.name](config=config, **dataConfig)

# Build a parent class for GTZANDataModuel and future DataModule (Not include FeautreDataModule)
class BaseAudioDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.GAIN_FACTOR = torch.tensor(0.11512925464970229)  # Example gain factor, adjust as needed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = None
        self.config = None
        self.data_dir = None
        self.batch_size = None
        self.num_workers = None
        self.required_key = None
        self.preprocessor_list = None # should be set in setup() by calling get_preprocessors()

    def get_preprocessors(self, preprocessors=None):
        # careful with the order
        preprocessor_dict = {
            'pad_or_truncate': self.pad_or_truncate,
            'resample': self.resample,
            'monolize': self.monolize,
            'normalize': self.normalize,
            'ensure_max_of_audio': self.ensure_max_of_audio,
        }
        preprocessor_list = []
        if not preprocessors:
            return preprocessor_list
        if isinstance(preprocessors, (list, omegaconf.listconfig.ListConfig)):
            for preprocessor in preprocessors:
                preprocessor_list.append(preprocessor_dict[preprocessor])
        elif isinstance(preprocessors, str):
            preprocessor_list.append(preprocessor_dict[preprocessors])
        else:
            raise TypeError(f'incompatible type for preprocessors of {preprocessors.__class__}')
        return preprocessor_list
    
    #################################
    # Lots of preprocessing methods #
    #################################
    def pad_or_truncate(self, audio):
        """
        audio: seq_len
        """
        max_length = self.config.data.max_length
        if audio.size(-1) > max_length:
            audio = audio[:, :max_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, max_length - audio.size(-1)))
        return audio

    def resample(self, audio): 
        orig_sample_rate = self.config.data.orig_sample_rate
        audio = torchaudio.functional.resample(audio, orig_sample_rate, self.sample_rate)
        return audio

    def monolize(self, audio):
        # make sure ouput shape is [1, seq_len]
        if len(audio.shape) == 2:
            if audio.shape[0] == 2:
                audio = audio.mean(dim=0, keepdim=True)
            elif audio.shape[0] == 1:
                pass
            else:
                audio = audio.mean(dim=1, keepdim=True).transpose(0, 1)
        else:
            audio = audio.unsqueeze(0)
        return audio

    def normalize(self, audio, db = -24):
        """Normalizes the signal's volume to the specified db, in LUFS.
        This is GPU-compatible, making for very fast loudness normalization.

        Parameters
        ----------
        db : typing.Union[torch.Tensor, np.ndarray, float], optional
            Loudness to normalize to, by default -24.0

        Returns
        -------
        AudioSignal
            Normalized audio signal.
        """
        ref_db = -24.0
        db = self.config.data.get('normalize_db') or db
        gain = db - ref_db
        gain = torch.exp(gain * self.GAIN_FACTOR)
        audio = audio * gain
        return audio

    def ensure_max_of_audio(self, audio, max_amplitude=1.0):
        """Ensures that ``abs(audio_data) <= max``.

        Parameters
        ----------
        max : float, optional
            Max absolute value of signal, by default 1.0

        Returns
        -------
        AudioSignal
            Signal with values scaled between -max and max.
        """
        max_amplitude = self.config.data.get('max_amplitude') or max_amplitude
        peak = audio.abs().max(dim=-1, keepdim=True)[0] # [batch, (channel), 1]
        peak_gain = torch.ones_like(peak) # [batch, channel, 1]
        peak_gain[peak > max_amplitude] = max_amplitude / peak[peak > max_amplitude] # [batch, (channel), 1]
        audio = audio * peak_gain  # [batch, (channel), seq_len]
        return audio
    
    
    def collate_fn(self, batch):
        waveforms, meta = [], {k:[] for k in self.required_key}
        for waveform, info in batch:
            for proc in self.preprocessor_list:
                waveform = proc(waveform)
            waveforms.append(waveform)
            # label preprocessing
            if 'label' in info and 'label' in self.required_key:
                meta['label'].append(info['label']) #label should be a integer scalar
            if 'beat_t' in info and 'beat_f' in self.required_key:
                meta['beat_f'].append(time_to_frame(info['beat_t'], fps=self.config.model.gen_model.fps, n_frame=None).clone().detach())

        # Wrap the collected data
        waveforms = torch.stack(waveforms)
        for k, v in meta.items():
            try:
                meta[k] = torch.stack(v)
            except:
                pass
        return waveforms, meta

    

class GTZANDataModule(BaseAudioDataModule):
    def __init__(self, config, data_dir: str, batch_size: int = 8, num_workers: int = 0, sample_rate: int = None, num_samples=650000, required_key=None, **kwargs):
        super().__init__()
        assert sample_rate is not None, 'Desired sample rate must be assigned manually'
        self.config = config
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.required_key = required_key
        self.setup()

    def prepare_data(self):
        GTZAN(root=self.data_dir, download=True)

    def setup(self, stage=None):
        self.train_dataset = GTZAN(root=self.data_dir, download=False, subset='training', folder_in_archive='genres_original')
        self.val_dataset = GTZAN(root=self.data_dir, download=False, subset='validation', folder_in_archive='genres_original')
        self.test_dataset = GTZAN(root=self.data_dir, download=False, subset='testing', folder_in_archive='genres_original')

        # Create label mapping
        self.preprocessor_list = self.get_preprocessors(self.config.data.get('preprocessors'))
        
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)


class GSDataModuel(BaseAudioDataModule):
    def __init__(self, config, data_dir: str, batch_size: int = 32, num_workers: int = 4, required_key=None, **kwargs):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.required_key = required_key
        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = GSDataset(root=self.data_dir, subset='train')
        self.val_dataset = GSDataset(root=self.data_dir, subset='valid')
        self.test_dataset = GSDataset(root=self.data_dir, subset='test')
        
        self.preprocessor_list = self.get_preprocessors(self.config.data.get('preprocessors'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    
    
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
    


class PreComputeDataModule(BaseAudioDataModule):
    def __init__(self, config, data_dir: str, batch_size: int = 32, num_workers: int = 4, required_key=None, **kwargs):
        super().__init__()
        # data_dir should contain train / valid / test 3 directories 
        self.config = config
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_to_index = {}
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
       # input should be in the shape of [.., layers, seq_len, dim]
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
            
class GSDataset(Dataset):
    """
    Sample data for ginatsteps_clip.json
    "0005061-0": {
        "clip": {
        "audio_duration": 120.058776,
        "audio_uid": "0005061",
        "clip_duration": 30.0,
        "clip_idx": 0,
        "clip_offset": 0.0
        },
        "extra": {
        "annotations": {
            "C": "2",
            "ID": "5061",
            "MANUAL KEY": "D# minor"
        },
        "beatport_metadata": {
            "ARTIST": "Philippe Van Mullem",
            "BP BPM": "132",
            "BP GENRE": "Trance",
            "BP KEY": "D# major",
            "ID": "5061",
            "LABEL": "Bonzai Classics",
            "MIX": "Progress Mix",
            "SONG TITLE": "Canopy"
        },
        "genre": "trance\n",
        "giantsteps.genre": "#@format: tag\ttimestamp(float)\tname(str)\t[weight(float)]\ntag\t0\ttrance\n",
        "giantsteps.key": "#@format: key\ttimestamp(float)\tkey(string)\t[confidence(int)]\t[comment(string)]\nkey\t0\td# minor\t2\t\n",
        "id": 5061,
        "key": "d# minor\t2\t\n"
        },
        "split": "train",
        "y": "Eb minor"
    },
    """
    def __init__(self, root, subset):
        super().__init__()
        self.root = root
        self.subset = subset
        # there are 24 classes in total
        self.classes = """C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major, C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor""".split(", ")
        self.class2id = {c: i for i, c in enumerate(self.classes)}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.min_bpm = 40
        self.max_bpm = 250
        self.metadata = pd.read_json(os.path.join(self.root, 'giantsteps_clips.json')).T
        # filter the split out
        self.metadata = self.metadata[self.metadata['split'] == self.subset]
        # self.metadata = self.metadata.to_dict(orient='records')
    
    def __getitem__(self, index):
        meta = self.metadata.iloc[index]
        aids = meta['extra']['id']
        audio, sr = torchaudio.load(
            os.path.join(self.root, 'audio', f"{aids}.LOFI.mp3"), 
            frame_offset=int(meta['clip']['clip_offset'] * 44100), 
            num_frames=int(meta['clip']['clip_duration'] * 44100)
        )
        tempo = (meta['extra']['beatport_metadata']['BP BPM'])
        if tempo == '' or tempo == None:
            tempo = 0
        else:
            tempo = float(tempo)
        key = meta['y']
        info = {'tempo': tempo, 'key': self.class2id[key], 'scaled_tempo': (tempo - self.min_bpm) / (self.max_bpm - self.min_bpm)}
        return audio, info
   
    def __len__(self):
        return len(self.metadata)
    
class MTGAudioDataset(Dataset):  # TODO: need to finish this part
    def __init__(self, root, subset,  required_key, split_version=0):  # TODO: incoporate split_version into configuration. Change low quality to high quality source audios.
        super().__init__()
        """
        Folder structure: 
        root -- mtg-jamendo-dataset -- data -- splits -- split-{split_version} -- autotagging_genre-{subset}.tsv
             -- data -- folder_number -- mp3files
        """
        if subset == 'valid':
            subset = 'validation'

        self.root = root
        self.metadata_dir = os.path.join(root, f'mtg-jamendo-dataset/data/splits/split-{split_version}/autotagging_genre-{subset}.tsv')
        
        self.split_version = split_version
        self.metadata = open(self.metadata_dir, 'r').readlines()[1:]

        self.all_paths = [line.split('\t')[3] for line in self.metadata]
        self.all_tags = [line.split('\t')[5:] for line in self.metadata]

        assert len(self.all_paths) == len(self.all_tags) == len(self.metadata)
        # set class2id
        self.set_class2id_dicts()
       

    def __getitem__(self, index):
        audio_path = os.path.join('data', self.all_paths[index])
        class_names = self.all_tags[index] # multiple tags
        audio, sr = torchaudio.load(os.path.join(self.root, audio_path))
        info = {'label': self.get_class2id(class_names)}
        
        return audio, info

    def __len__(self):
        return len(self.metadata)
    
    def set_class2id_dicts(self):
        class2id_dict = {}
        for subset in ['train', 'validation', 'test']:
            data = open(os.path.join(self.root, f'mtg-jamendo-dataset/data/splits/split-{self.split_version}/autotagging_genre-{subset}.tsv'), "r").readlines()
            for example in data[1:]:
                tags = example.split('\t')[5:]
                for tag in tags:
                    tag = tag.strip()
                    if tag not in class2id_dict:
                        class2id_dict[tag] = len(class2id_dict)
        self.class2id = class2id_dict
        self.id2class = {v: k for k, v in self.class2id.items()}
    
    def get_class2id(self, class_names):
        # return the multi-hot encoding of the class names
        classid = torch.zeros(len(self.class2id))
        for class_name in class_names:
            if class_name in self.class2id:
                class_id = self.class2id[class_name]
                classid[class_id] = 1
        return classid
    
  
    
class MTGDataModule(BaseAudioDataModule):  # TODO: need to finish this part
    def __init__(self, config, data_dir: str, batch_size: int = 32, num_workers: int = 4, sample_rate: int = None, num_samples=650000, required_key=None, **kwargs):  # TODO: incoporate split_version into configuration. Change low quality to high quality source audios.
        super().__init__()
        # TODO: this path has to be input from the config file
        self.config = config
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_to_index = {}
        self.required_key = required_key
        self.setup()
    
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = MTGAudioDataset(root=self.data_dir, subset='train', required_key=self.required_key)
        self.val_dataset = MTGAudioDataset(root=self.data_dir, subset='validation', required_key=self.required_key)
        self.test_dataset = MTGAudioDataset(root=self.data_dir, subset='test', required_key=self.required_key)

        self.preprocessor_list = self.get_preprocessors(self.config.data.get('preprocessors'))
        
    
    # def collate_fn(self, batch):
    #     waveforms, meta = [], dict(label=[], beat_t=[], beat_f=[]),
    #     for waveform, info in batch:
    #         # waveform preprocessing
    #         waveform = self.pad_or_truncate(waveform)
    #         waveform = torchaudio.functional.resample(waveform, info['sample_rate'], self.sample_rate)
    #         waveforms.append(waveform)
    #         # label preprocessing
    #         meta['label'].append(torch.tensor(self.label_to_index[info['label']]))
    #         meta['beat_f'].append(time_to_frame(info['beat_t'], fps=self.config.model.gen_model.fps, n_frame=None).clone().detach())
    #     # Wrap the collected data
    #     waveforms = torch.stack(waveforms)
    #     for k, v in meta.items():
    #         try:
    #             meta[k] = torch.stack(v)
    #         except:
    #             pass
    #     return waveforms, meta
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
        
        

if __name__ == '__main__':
    from omegaconf import OmegaConf
    def testMTG():
        testconf = {
            'data': {
                'name': 'mtg',
                'data_dir': '../../Database/MTG/',
                'batch_size': 4,
                'num_workers': 0,
                'orig_sample_rate': 44100,
                'sample_rate': 44100,
                'max_length': 44100 * 25,
                'preprocessors': ['pad_or_truncate', 'resample', 'monolize', 'normalize', 'ensure_max_of_audio'],
                'normalize_db': -24,
                'max_amplitude': 1.0,
            },
            'model': {
                'gen_model': {
                    'fps': 100,
                    'output_dim': 1280
                }
            }
        }
        testconf = OmegaConf.create(testconf)
        print('start creating datamodule')
        dm = MTGDataModule(config=testconf, data_dir=testconf.data.data_dir, batch_size=testconf.data.batch_size, num_workers=testconf.data.num_workers, sample_rate=testconf.data.sample_rate, required_key=['label'])
        print('start setting up')
        dls = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]
        print('start iterating')
        for dl in dls:
            for data in dl:
                waveforms, meta = data
                print(f"Waveforms shape: {waveforms.shape}")
                for k, v in meta.items():
                    print(f"{k} shape: {v.shape}")
                    break
                break
            break
        
    def testGTZAN():
        testconf = {
            'data': {
                'name': 'genre_classification',
                'data_dir': '../scratch/GTZAN/raw',
                'batch_size': 4,
                'num_workers': 0,
                'sample_rate': 32000,
            },
            'model': {
                'gen_model': {
                    'max_length': 22050 * 25,
                    'sample': 22050,
                    'fps': 100,
                    'preprocessors': ['pad_or_truncate', 'resample', 'monolize', 'normalize', 'ensure_max_of_audio'],
                    'normalize_db': -24,
                    'max_amplitude': 1.0,
                }
            }
        }
        testconf = OmegaConf.create(testconf)
        dm = GTZANDataModule(config=testconf, data_dir=testconf.data.data_dir, batch_size=testconf.data.batch_size, num_workers=testconf.data.num_workers, sample_rate=testconf.data.sample_rate, required_key=['label', 'beat_f'])
        dm.setup()
        dls = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]
        for dl in dls:
            for data in dl:
                waveforms, meta = data
                print(f"Waveforms shape: {waveforms.shape}")
                for k, v in meta.items():
                    print(f"{k} shape: {v.shape}")
                    """
                    Waveforms shape: torch.Size([4, 1, 1102500])
                    label shape: torch.Size([4, 87])
                    """
                    break
                break
            break

    testMTG()