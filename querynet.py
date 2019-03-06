from argparse import ArgumentParser
import pickle

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import Sequence
from keras.layers import Dense, Embedding, Input, Concatenate, Reshape
from keras.models import Model, load_model
from sklearn.decomposition import PCA


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
        np.random.seed(idx)
        query_points_x = np.random.randint(0, self.data_array.shape[1], self.batch_size)
        query_points_y = np.random.randint(0, self.data_array.shape[2], self.batch_size)
        color_values = self.data_array[idx, 
                                       query_points_x,
                                       query_points_y,
                                       :]
        query_coordinates = np.stack([query_points_x, query_points_y], -1)
        
        #put all images between zero and one:
        query_coordinates = query_coordinates / self.data_array.shape[1:3]

        return ({'coordinates': query_coordinates,
                 'idx': np.repeat(idx, self.batch_size)},
                color_values)


def create_representation(num_images: int, embedding_dim: int):
    idx = Input(shape=(None, ), name='idx')
    embedded_image_input = Embedding(num_images, embedding_dim, input_length=1)(idx)
    embedded_image_input = Reshape([-1])(embedded_image_input)

    return Model(inputs=idx, outputs=embedded_image_input)


def create_render_network(embedding_dim: int):
    coordinates = Input(shape=[2], name='coordinates')
    latent_vec = Input(shape=[embedding_dim], name='latent_vec')
    x = Concatenate(axis=-1)([coordinates, latent_vec])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(62, activation='relu')(x)
    x = Concatenate(axis=-1)([coordinates, x])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(3, activation='sigmoid')(x)

    return Model(inputs=[coordinates, latent_vec],
                 outputs=x)


def make_models(num_images: int, embedding_dim: int):
    rep = create_representation(num_images, embedding_dim)
    render = create_render_network(embedding_dim)
    coordinates = Input(shape=[2], name='coordinates')
    
    output = render([coordinates, rep.output])
    return Model(inputs=[coordinates, rep.input], outputs=output), render


def image_from_idx(model, idx, image_size=(32, 32)):
    coordinates = np.array(list(np.ndindex(*image_size)))
    q_coordinates = coordinates / image_size

    y = model.predict_on_batch({'coordinates': q_coordinates,
                 'idx': np.repeat(idx, len(q_coordinates))})
    
    img = np.zeros([*image_size, 3])    
    for c, v in zip(coordinates, y):
        img[c[0],c[1] :] = v      
    return img


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train_data',
                        default='data/cifar-10-batches-py/data_batch_1',
                        type=str)
    parser.add_argument('--validation_data',
                        default='data/cifar-10-batches-py/data_batch_5',
                        type=str)
    parser.add_argument('--test_data',
                        default='data/cifar-10-batches-py/test_batch',
                        type=str)
    parser.add_argument('--model_path',
                        default='trained_models/querynet.h5',
                        type=str)
    parser.add_argument('--limit_images',
                        default=None,
                        type=int)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--steps_per_epoch', default=None, type=int)
    return parser.parse_args()


def encoder_plot(model, data):
    for i in range(10):
        plt.figure('GT')
        plt.subplot(5, 2, i + 1)
        plt.imshow(image_from_idx(model, i))
        plt.figure('Pred')
        plt.subplot(5, 2, i + 1)
        plt.imshow(data[i, ...])


def embedding_plot(embeddings, labels):
    pca = PCA(n_components=2)
    plt.figure("PCA")
    X = pca.fit_transform(embeddings)
    print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', alpha=0.3)


def auto_decode(data: np.ndarray, model: Model, dim, samples: int = 1024):
    generator = DataGenerator(data, samples)

    # freeze the representation network
    for layer in model.layers:
        layer.trainable = False

    # add a new embedding layer
    rep = create_representation(len(data), dim)
    coordinates = Input(shape=[2], name='coordinates')

    output = model([coordinates, rep.output])
    full = Model(inputs=[coordinates, rep.input], outputs=output)
    full.compile('adam', loss='mean_squared_error')
    full.fit_generator(generator, epochs=10, steps_per_epoch=10)
    plt.figure('Auto decoded')
    plt.subplot(211)
    plt.imshow(image_from_idx(full, 0, image_size=(512, 512)))
    plt.subplot(212)
    plt.imshow(data[0])


def main():
    args = parse_args()
    data, labels = load_batch(args.train_data)
    t_data, t_labels = load_batch(args.test_data)

    if args.load:
        model = load_model(args.model_path)
    else:
        train_generator = DataGenerator(data[:args.limit_images, :])

        model, render = make_models(len(train_generator), 30)
        model.compile('adam', loss='mean_squared_error')
        model.fit_generator(train_generator,
                            epochs=args.epochs,
                            steps_per_epoch=args.steps_per_epoch)
        
        model.save(args.model_path)

    encoder_plot(model, data)
    
    embedding_plot(model.get_layer('embedding_1').get_weights()[0],
            labels[:args.limit_images])
    plt.figure('super sample')
    plt.imshow(image_from_idx(model, 0, image_size=(512, 512)))
    auto_decode(data[[0], ...], render, 30, samples=100)
    plt.show()


if __name__ == '__main__':
    main()
