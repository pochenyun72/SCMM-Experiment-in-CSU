# coding: utf-8

# 导包
from keras.datasets import mnist
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from keras.models import Sequential, Model

# 1.数据预处理
# 1.1 加载训练集和测试集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# 1.2 重塑训练集和测试集的形状
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype(dtype='float32')
X_test = X_test.astype(dtype='float32')

print(X_train.shape)
print(X_test.shape)
print(X_train.dtype)
print(X_test.dtype)

# 1.3 归一化,从0~255的取值压缩到0~1之间
X_train /= 255
X_test /= 255

# 1.4 one-hot编码
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 2. 搭建神经网络
# 2.1 添加层
model = Sequential()
model.add(Dense(10, input_shape=(784,)))
model.add(Activation('softmax'))

model.summary()

# 2.2 编译神经网络
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

# 2.3 训练神经网络
history = model.fit(X_train, y_train, batch_size=128, epochs=200, verbose=1, validation_split=0.2)

# 2.4 评估神经网络
score = model.evaluate(X_test, y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# 3. 优化神经网络
# 3.1 模型改进
model2 = Sequential()
model2.add(Dense(128, input_shape=(784,)))
model2.add(Activation('softmax'))
model2.add(Dense(128))
model2.add(Activation('relu'))
model2.add(Dense(10))
model2.add(Activation('softmax'))

model2.summary()

# 3.2 编译神经网络
model2.compile(loss='categorical_crossentropy',
               optimizer='SGD',
               metrics=['accuracy'])

# 3.3 训练神经网络
history = model2.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2)

# 3.4 评估神经网络
score = model2.evaluate(X_test, y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# 4. 模型优化2
# 4.1 模型改进
model3 = Sequential()
model3.add(Dense(512, input_shape=(784,)))
model3.add(Activation('relu'))
model3.add(Dropout(0.2))
model3.add(Dense(512))
model3.add(Activation('relu'))
model3.add(Dropout(0.2))
model3.add(Dense(10))
model3.add(Activation('softmax'))

model3.summary()

# 4.2 编译神经网络
model3.compile(loss='categorical_crossentropy',
               optimizer='rmsprop',
               metrics=['accuracy'])

# 4.3 训练神经网络
history = model3.fit(X_train, y_train, epochs=10, batch_size=128,
                     verbose=1, validation_data=[X_test, y_test])

# 4.4 评估神经网络
score = model3.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
