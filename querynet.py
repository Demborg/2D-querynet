from argparse import ArgumentParser
import pickle

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import Sequence

def load_batch(path: str):
    data_dict = pickle.load(open(path, 'rb'), encoding='bytes')
    image_data = np.reshape(data_dict[b'data'], [-1, 3, 32, 32])
    image_data = np.moveaxis(image_data, 1, -1)
    return image_data / 255, data_dict[b'labels']

class DataGenerator(Sequence):
    def __init__(self, data_array, batch_size=32**2):
        self.data_array = data_array
        self.batch_size = batch_size

    def __len__(self):
        return self.data_array.shape[0]

    def __getitem__(self, idx):
        query_points_x = np.random.randint(0, self.data_array.shape[1], self.batch_size)
        query_points_y = np.random.randint(0, self.data_array.shape[2], self.batch_size)
        color_values = self.data_array[idx, 
                                       query_points_x,
                                       query_points_y,
                                       :]
        query_coordinates = np.stack([query_points_x, query_points_y], -1)
        return ({'coordinates': query_coordinates,
                 'idx': np.repeat(idx, self.batch_size)},
                color_values)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train_data',
                        default='data/cifar-10-batches-py/data_batch_1',
                        type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    data, _  = load_batch(args.train_data)
    gen = DataGenerator(data, 4)
    l = gen.__getitem__(0)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
