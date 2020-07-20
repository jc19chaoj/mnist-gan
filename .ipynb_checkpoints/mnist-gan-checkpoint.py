import sys
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
print('Tensorflow version:', tf.__version__)

def download_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    return x_train, x_test

def train_dcgan(gan, dataset, batch_size, num_features, seed, epochs=5):
    generator, discriminator = gan.layers
    for epoch in tqdm(range(epochs)):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        for X_batch in dataset:
            noise = tf.random.normal(shape=[batch_size, num_features])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            noise = tf.random.normal(shape=[batch_size, num_features])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)

        #display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)
        
    #display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)
    
def generate_and_save_images(model, epoch, test_input):
    
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10,10))

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='binary')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

    
def main(argv):
    
    # parse arguments
    args = parser.parse_args(argv[1:])
    batch_size = args.batch_size
    epochs = args.epochs
    
    # Create generator and discriminator models
    num_features = 100

    generator = keras.models.Sequential([
        keras.layers.Dense(7 * 7 * 128, input_shape=[num_features]),
        keras.layers.Reshape([7, 7, 128]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(128, (5,5), (1,1), padding="same", activation="selu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(64, (5,5), (2,2), padding="same", activation="selu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(1, (5,5), (2,2), padding="same", activation="tanh"),
    ])

    discriminator = keras.models.Sequential([
        keras.layers.Conv2D(64, (5,5), (2,2), padding="same", input_shape=[28, 28, 1]),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(128, (5,5), (2,2), padding="same"),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Conv2D(256, (5,5), (1,1), padding="same"),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dropout(0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
    discriminator.trainable = False
    gan = keras.models.Sequential([generator, discriminator])
    gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
    
    print("[INFO] Downloading dataset...")
    x_train, x_test = download_dataset()
    print("... Complete!")
    
    seed = tf.random.normal(shape=[batch_size, 100])
    x_train_dcgan = x_train.reshape(-1, 28, 28, 1) * 2. - 1.
    
    
    print("[INFO] Shuffling training data, batch size={}...".format(batch_size), end='')
    dataset = tf.data.Dataset.from_tensor_slices(x_train_dcgan)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    print("... Complete!")
    
    print("[INFO] Start training, num of epochs={}...".format(epochs))
    train_dcgan(gan, dataset, batch_size, num_features, seed, epochs=epochs)
    print("[SUCCESS] Training completed!")
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=5, type=int, help='number of epochs')
    main(sys.argv)
