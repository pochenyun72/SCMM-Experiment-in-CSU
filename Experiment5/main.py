import numpy as np
import pandas as pd
import os

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
path = '/home/pochenyun/Projects/VScode/SC&MM/Experiment5/data/mnist.npz'
f = np.load(path)
train_images, train_labels = f['x_train'], f['y_train']
test_images, test_labels = f['x_test'], f['y_test']
f.close()
from keras.utils.np_utils import to_categorical

# reshape
X_train = X_train.reshape([60000, 784])
X_test = X_test.reshape([10000, 784])
X_train.shape, X_test.shape
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_train = X_train / 255
X_test = X_test / 255
print(X_train)
train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(10, input_shape=(28 * 28,)))
model.add(Activation('softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# summary
model.summary()  # fit
model.fit(X_train, train_labels, epochs=200, batch_size=128, verbose=True, validation_split=0.2)
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=True)
print('test_loss', test_loss, '\n', 'test_acc', test_acc)
model = Sequential()
model.add(Dense(128, input_shape=(28 * 28,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# summary
model.summary()
model.fit(X_train, train_labels, epochs=200, batch_size=128, verbose=True, validation_split=0.2)
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=True)
print('test_loss', test_loss, '\n', 'test_acc', test_acc)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
from keras.layers import Conv2D, BatchNormalization, Dropout, Conv2D, MaxPool2D, Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu',
                 input_shape=(28, 28, 1)))
# add two layers => avoid overfit
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.5))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu',
                 input_shape=(28, 28, 1)))
model.add(BatchNormalization(axis=1))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# summary
model.summary()
model.fit(X_train, train_labels, epochs=15, batch_size=128, verbose=True, validation_split=0.1)
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=True)
print('test_loss', test_loss, '\n', 'test_acc', test_acc)
from keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.10,
                               shear_range=0.5, height_shift_range=0.10, zoom_range=0.10)
# compile model with data generator
model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.optimizer.lr = 0.001
batches = generator.flow(X_train, train_labels, batch_size=128)
history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=
15)
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=True)
print('test_loss', test_loss, '\n', 'test_acc', test_acc)
img_size = 28
img_size_flat = 28 * 28
img_shape = (28, 28)
img_shape_full = (28, 28, 1)
num_classes = 10
num_channels = 1
import matplotlib.pyplot as plt


# plot_some images
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == 9
    assert len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = 'True:{0}'.format(cls_true[i])
        else:
            xlabel = 'True:{0}, Pred:{1}'.format(cls_true[i], cls_pred[i])
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


# plot single image
def plot_image(image):
    plt.imshow(image.reshape(img_shape), interpolation='nearest', cmap='binary')
    plt.show()


# plot error example
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = X_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = y_test[incorrect]
    # plot 0:9
    plot_images(images=images[0:9], cls_true=cls_true[0:9], cls_pred=cls_pred[0:9])


from keras.models import Sequential
from keras.layers import InputLayer, Reshape, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
# build model
model.add(InputLayer(input_shape=(img_size_flat,)))
model.add(Reshape(img_shape_full))
model.add(Conv2D(kernel_size=5, strides=1, filters=36,
                 padding='same', activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Conv2D(kernel_size=5, strides=1, filters=36,
                 padding='same', activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
model.fit(X_train, train_labels, epochs=1, batch_size=128, validation_split=1 / 12, verbose=True)
result = model.evaluate(X_test, test_labels, verbose=True)
print('loss ', result[0])
print('acc ', result[1])
predict = model.predict(X_test)
predict = np.argmax(predict, axis=1)
plot_images(X_test[0:9], y_test[0:9], predict[0:9])
y_pred = model.predict(X_test)
cls_pred = np.argmax(y_pred, axis=1)
correct = (cls_pred == y_test)
plot_example_errors(cls_pred, correct=correct)
from keras.models import Model
from keras.layers import Input, Reshape, Conv2D, MaxPooling2D, Flatten, Dense
from keras import backend as K

inputs = Input(shape=(img_size_flat,))
net = inputs
net = Reshape(img_shape_full)(net)
net = Conv2D(kernel_size=5, strides=1, filters=16,
             padding='same', activation='relu', name='layer_conv1')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = Conv2D(kernel_size=5, strides=1, filters=36,
             padding='same', activation='relu', name='layer_conv2')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = Flatten()(net)
net = Dense(128, activation='relu')(net)
net = Dense(num_classes, activation='softmax')(net)
outputs = net
# then use keras.models.Model to build model
model2 = Model(inputs=inputs, outputs=outputs)
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])
# summary
model2.summary()
model2.fit(X_train, train_labels, batch_size=128, epochs=1,
           validation_split=1 / 12, verbose=True)
result = model2.evaluate(X_test, test_labels, verbose=True)
print(model2.metrics_names[0], result[0])
print(model2.metrics_names[1], result[1])
predict = model2.predict(X_test)
predict = np.argmax(predict, axis=1)
plot_images(X_test[0:9], y_test[0:9], predict[0:9])
y_pred = model.predict(X_test)
cls_pred = np.argmax(y_pred, axis=1)
correct = (cls_pred == y_test)
plot_example_errors(cls_pred, correct=correct)
from keras.models import load_model
import math

path_model = 'model2.pkl'
model2.save(path_model)
# del model
del model2
# load mdoel
model3 = load_model('model2.pkl')
model.summary()
predict = model3.predict(X_test)
predict = np.argmax(predict, axis=1)
plot_images(X_test[0:9], y_test[0:9], predict[0:9])


def plot_conv_weights(weights, input_channel=0):
    # get params
    w_min = np.min(weights)
    w_max = np.max(weights)
    num_filters = weights.shape[3]
    # plot-prework
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = weights[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# output conv-output
def plot_conv_output(values):
    # get params
    num_filters = values.shape[3]
    # plot-prework
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

model3.summary()
layer_input = model3.layers[0]
layer_conv1 = model3.layers[2]
layer_conv2 = model3.layers[4]
# get weights
weights_conv1 = layer_conv1.get_weights()[0]
plot_conv_weights(weights=weights_conv1, input_channel=0)
weights_conv2 = layer_conv2.get_weights()[0]
plot_conv_weights(weights=weights_conv2, input_channel=0)
image1 = X_test[0]
plot_image(image1)
output_conv1 = K.function(inputs=[layer_input.input], outputs=[layer_conv1.output])
layer_output1 = output_conv1(np.array([image1]))[0]
print(layer_output1.shape)
plot_conv_output(values=layer_output1)
output_conv2 = Model(inputs=layer_input.input, outputs=layer_conv2.output)
layer_output2 = output_conv2.predict(np.array([image1]))
print(layer_output2.shape)
plot_conv_output(values=layer_output2)
