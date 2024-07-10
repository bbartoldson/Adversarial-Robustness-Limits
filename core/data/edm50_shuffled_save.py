import os
import h5py
import numpy as np


def get_shuffled_EDM_50_indices():
    indices_path = "/p/vast1/MLdata/virtual/EDM_50_indices.npy"
    if not os.path.exists(indices_path):
        EDM_50_indices = np.arange(int(50e6))
        np.random.shuffle(EDM_50_indices)
        np.save(indices_path, EDM_50_indices)
        return EDM_50_indices
    return np.load(indices_path)

def main(name):
    shuffled_indices = get_shuffled_EDM_50_indices()
    
    print('Loading data. This might take a while.')
    aux = np.load("/p/vast1/MLdata/CIFAR-10-EDM/50m.npz")
    shuffled_aux_image = aux['image'][shuffled_indices]
    shuffled_aux_label = aux['label'][shuffled_indices]

    print('Loaded and shuffled data.')

    assert len(shuffled_aux_image) == len(shuffled_aux_label) == 50e6

    print(f'Orig labels       { aux["label"][:5]}')
    print(f'Shuffled labels   {shuffled_aux_label[:5]}')
    print(f'Shuffled indices  {shuffled_indices[:5]}')
    print(f'\nImage and label shapes are {shuffled_aux_image.shape} and {shuffled_aux_label.shape}')

    if not os.path.exists(name):
        with h5py.File(name=name, mode='w') as f:
            image = f.create_dataset('image', shuffled_aux_image.shape, dtype=np.uint8)
            image[:] = shuffled_aux_image
            label = f.create_dataset('label', shuffled_aux_label.shape, dtype=np.uint8)
            label[:] = shuffled_aux_label
    else:
        assert False, f'The file {name} exists! Skipping creation.'

if __name__=='__main__':
    name = '/p/vast1/MLdata/virtual/edm50_shuffled.h5'  
    if not os.path.exists(name):
        main(name)
    else:
        print(f'The file {name} exists! Skipping creation.')
    