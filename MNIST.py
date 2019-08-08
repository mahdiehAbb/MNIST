import keras
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Sequential, Model
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import tensorflow as tf
from keras.constraints import MinMaxNorm

def save_model_json(model, file_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(file_name + '.json', "w") as json_file:
        json_file.write(model_json)

def save_model_weights(model, file_name):
    # serialize weights to HDF5
    model.save_weights(file_name + '.h5')


def save_model_json_h5(model, file_name):
    save_model_json(model, file_name)
    save_model_weights(model, file_name)
    print('Saved model to disk')

def save_model(model, file_name):
    # Train.save_model_json(model, file_name)
    # Train.save_model_hdf5(model, file_name)
    model.save(file_name + '.h5')
    print('Saved model to disk')


# Preparing the dataset
# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Making a copy before flattening for the next code-segment which displays images
x_train_drawing = x_train

image_size = 784  # 28 x 28
x_train = x_train.reshape(x_train.shape[0], image_size).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], image_size).astype('float32') / 255

# # Making sure that the values are float so that we can get decimal points after division
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# # Normalizing the RGB codes by dividing it to the max RGB value.
# x_train /= 255
# x_test /= 255



# some_digits = [0, 1]
# train_some_digits = [i for i, value in enumerate(y_train) if value in some_digits]
# y_train_some_digits = y_train[train_some_digits]
# x_train_some_digits = x_train[train_some_digits]
# count_train_0 = np.count_nonzero(y_train_some_digits == 0)
# count_train_1 = np.count_nonzero(y_train_some_digits == 1)
#
#
# test_some_digits = [i for i, value in enumerate(y_test) if value in some_digits]
# y_test_some_digits = y_test[test_some_digits]
# x_test_some_digits = x_test[test_some_digits]
# count_test_0 = np.count_nonzero(y_test_some_digits == 0)
# count_test_1 = np.count_nonzero(y_test_some_digits == 1)

# np.savetxt('inputs.csv', x_test_some_digits, fmt='%f', delimiter=',')
# np.savetxt('labels.csv',y_test_some_digits, fmt='%d', delimiter=',')

# Convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# model = Sequential()
# # The input layer requires the special input_shape parameter which should match
# # the shape of our training data.
# # MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0
# model.add(Dense(units=4, activation='sigmoid', input_shape=(image_size,), use_bias=False, kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0)))
# model.add(Dense(units=num_classes, activation='softmax', use_bias=False, kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0)))
# model.summary()

inputs = Input(shape=(784,), name='img')

dense_1 = Dense(90, activation='relu', use_bias=False, kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0))
intermediate_output = dense_1(inputs)
dense_2 = Dense(60, activation='relu', use_bias=False, kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0))
intermediate_output = dense_2(intermediate_output)

dense = Dense(num_classes, activation="softmax", use_bias=False, kernel_constraint=MinMaxNorm(min_value=-1.0, max_value=1.0, rate=1.0, axis=0))
outputs = dense(intermediate_output)

intermediate_model = Model(inputs=inputs, outputs=intermediate_output)
model = Model(inputs=inputs, outputs=outputs, name='mnist_model')
model.summary()


logger = keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.1,
                    batch_size=128, epochs=400, verbose=2)
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(accuracy)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.savefig('myfilename.png')
plt.show()

# intermediate_sigmoid_predicted = intermediate_model.predict(x_test)
# np.savetxt('2_node_sigmoid_predicted.csv', intermediate_sigmoid_predicted, delimiter=',')
#
#
# predicted = model.predict(x_test, batch_size=1)
# np.savetxt('2_node_softmax_predicted.csv', predicted, delimiter=',')

save_model(model, '10_digits_2_hidden_layers_90_60')

converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file('10_digits_2_hidden_layers_90_60.h5')
# converter.optimizations = [tf.contrib.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()
open("10_digits_2_hidden_layers_90_60.tflite", "wb").write(tflite_model)
print('Converted model to the tflite format')

# interpreter = tf.contrib.lite.Interpreter(model_path="some_digits_2_model.tflite")
# interpreter.allocate_tensors()
#
# # Print input shape and type
# print(interpreter.get_input_details()[0]['shape'])  # Example: [1 224 224 3]
# print(interpreter.get_input_details()[0]['dtype'])  # Example: <class 'numpy.float32'>
#
# # Print output shape and type
# print(interpreter.get_output_details()[0]['shape'])  # Example: [1 1000]
# print(interpreter.get_output_details()[0]['dtype'])  # Example: <class 'numpy.float32'>








