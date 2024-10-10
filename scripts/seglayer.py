import os
import pickle
import h5py
import numpy as np
from tqdm import tqdm
import torch

def process_directory(input_dir, output_file):
    with h5py.File(output_file, 'w') as hf:
        for filename in tqdm(os.listdir(input_dir)):
            if filename.endswith('.pkl'):
                input_path = os.path.join(input_dir, filename)
                
                # with open(input_path, 'rb') as f:
                #     data = pickle.load(f)
                data = torch.load(input_path)
                
                repr_data = data['repr']  # Shape: [layers, seq_len, dim]
                meta_data = data['meta']
                
                # Create a group for each file
                file_group = hf.create_group(filename[:-4])  # Remove .pkl extension
                
                # Store repr data
                file_group.create_dataset('repr', data=repr_data, compression='gzip')
                
                # Store meta data
                meta_group = file_group.create_group('meta')
                for key, value in meta_data.items():
                    if isinstance(value, (str, int, float, bool)):
                        meta_group.attrs[key] = value
                    elif isinstance(value, (list, np.ndarray, torch.Tensor)):
                        if key == 'beat_t':
                            value = np.array(value)
                            meta_group.create_dataset(key, data=value, compression='gzip')
                        elif key == 'label':
                            value = value.detach().cpu()
                            meta_group.create_dataset(key, data=value)
                        else:
                            value = value.detach().cpu()
                            meta_group.create_dataset(key, data=value, compression='gzip')
                    else:
                        # For more complex types, you might need to serialize them
                        meta_group.attrs[key] = str(value)

def main():
    base_input_dir = '/home/hice1/ywu3038/scratch/GTZAN/MusicGenSmall_origin'
    base_output_dir = '/home/hice1/ywu3038/scratch/GTZAN/MusicGenSmall'
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    # for split in ['train', 'valid', 'test']:
    for split in ['valid']: 
        print(f"Processing {split} directory...")
        input_dir = os.path.join(base_input_dir, split)
        output_file = os.path.join(base_output_dir, f'{split}.h5')
        process_directory(input_dir, output_file)
    
    print("Processing complete!")

if __name__ == '__main__':
    main()