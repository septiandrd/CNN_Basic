from keras.layers import Dense, Conv2D, Conv2DTranspose, Activation, BatchNormalization, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
np.set_printoptions(suppress=True)
from PIL import Image
from scipy.misc import imresize,imshow
import matplotlib.pyplot as plt

def build_generator() :

    model = Sequential()
    model.add(Conv2DTranspose(64,(6,6),strides=2,padding='same',input_shape=(54,44,3)))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(64,(6,6),strides=2,padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(5,5),padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Conv2D(3,(5,5),padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('tanh'))

    model.summary()

    return model

def build_discriminator() :

    model = Sequential()
    model.add(Conv2D(64,(4,4),strides=(2,2),padding='same',input_shape=(216,176,3)))
    model.add(LeakyReLU())
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(BatchNormalization(axis=1))

    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))

    model.summary()

    return model


if __name__ == '__main__':

    optimizer = Adam(lr=2e-4,beta_1=0.5)
    epoch = 20

    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy',optimizer=optimizer)

    train_datagen = ImageDataGenerator()

    # train_generator = train_datagen.flow_from_directory('dataset',batch_size=2,target_size=(54,44))
    train_discriminator = train_datagen.flow_from_directory('dataset',batch_size=2,target_size=(216,176))

    # train_generator.next()
    #
    # print(train_generator.next())

    for epoch in range(1) :

        img_G = []
        img_D = train_discriminator.next()[0]
        for img in img_D :
            img_G.append(imresize(img,(54,44,3)))
        img_G = np.array(img_G)
        # print(np.shape(imgs))

        img_G = (img_G.astype(np.float32)-127.5) / 127.5
        print(img_G)

        gen_imgs = generat








        sdasdasdasw    or.predict(img_G)
        print(gen_imgs)
        gen_imgs = ((gen_imgs*1000+1)/2)*255
        print(gen_imgs)
        #
        # print(gen_imgs[1])
        plt.figure(1)
        plt.subplot(211)
        plt.imshow(np.uint8(img_D[0]))
        plt.subplot(212)
        plt.imshow(np.uint8(gen_imgs[0]))
        plt.show()

        d_loss_real = discriminator.train_on_batch(img_D,np.ones((2,1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs,np.ones((2,1)))
        d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)
        #
        #
        print('Epoch',epoch,':',d_loss)
    #

