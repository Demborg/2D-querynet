from argparse import ArgumentParser
import pickle

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import Sequence
from keras.layers import Dense, Embedding, Input, Concatenate, Reshape
from keras.models import Model

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


def create_network(num_images: int, embedding_dim: int):
    coordinates = Input(shape=[2])
    embedding_input = Input(shape=(None, ))
    embedded_image_input = Embedding(num_images, embedding_dim, input_length=1)(embedding_input)
    embedded_image_input = Reshape([-1])(embedded_image_input)
    x = Concatenate(axis=-1)([coordinates, embedded_image_input])
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(3, activation='sigmoid')(x)

    return Model(inputs = [coordinates, embedding_input],
                 outputs = x)
       

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train_data',
                        default='data/cifar-10-batches-py/data_batch_1',
                        type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    data, _  = load_batch(args.train_data)
    generator = DataGenerator(data)
    net = create_network(len(generator), 20)

if __name__ == '__main__':
    main()
