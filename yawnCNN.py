from __future__ import absolute_import
from __future__ import print_function

'''import tensorflow as tf
print(tf.__version__)
tf.test.is_gpu_available()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

'''
import numpy as np

np.random.seed(1337)  # for reproducibility 1337
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam

from sklearn import svm

from six.moves import cPickle as pickle
pickle_files = ['yawn_mouths.pickle']
i = 0
for pickle_file in pickle_files:
    with open(pickle_file,'rb') as f:
        save = pickle.load(f)
        if i == 0:
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
        else:
            train_dataset = np.concatenate((train_dataset, save['train_dataset']))
            train_labels = np.concatenate((train_labels, save['train_labels']))
            test_dataset = np.concatenate((test_dataset, save['test_dataset']))
            test_labels = np.concatenate((test_labels, save['test_labels']))
        del save  # hint to help gc free up memory
    i += 1

print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 1
nb_classes = 1
epochs = 50

X_train = train_dataset
X_train = X_train.reshape((X_train.shape[0], X_train.shape[3]) + X_train.shape[1:3])
# X_train = X_train.reshape(X_train.shape[0:3])
# X_train = X_train.reshape(len(X_train), -1)
Y_train = train_labels

X_test = test_dataset
X_test = X_test.reshape((X_test.shape[0], X_test.shape[3]) + X_test.shape[1:3])
# X_test = X_test.reshape(X_test.shape[0:3])
# X_test = X_test.reshape(len(X_test), -1)
Y_test = test_labels

# input image dimensions
_, img_channels, img_rows, img_cols = X_train.shape

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, (3, 3), padding='same',input_shape=(img_channels, img_rows, img_cols),data_format='channels_first'))
model.add(Activation('relu'))
'''
#original
model.add(Convolution2D(32, (3, 3),data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3),data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

'''

# face detection cnn layers
model.add(MaxPooling2D(pool_size=(2, 2),padding='same' ))

model.add(Convolution2D(36, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Convolution2D(48, 3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3),data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3),data_format='channels_first'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

'''
#another model yet to be tried
model = Sequential()

    model.add(Convolution2D(16, 5, 5, border_mode="full", input_shape=(3, height, width)))
    model.add(Convolution2D(16, 5, 5))
    model.add(Activation("relu"))
    model.add(Convolution2D(32, 5, 5))
    model.add(Activation("relu"))
    model.add(Convolution2D(32, 5, 5))
    model.add(Activation("relu"))
    model.add(Convolution2D(128, 5, 5))
    model.add(Activation("relu"))

    model.add(MaxPooling2D((2,2))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 5, 5))
    model.add(Activation("relu"))
    model.add(Convolution2D(256, 5, 5))
    model.add(Activation("relu"))
    model.add(Convolution2D(256, 5, 5))
    model.add(Activation("relu"))

    model.add(MaxPooling2D((2,2))
    model.add(Dropout(0.25))

    model.add(Convolution2D(265, 5, 5))
    model.add(Activation("relu"))
    model.add(Convolution2D(265, 5, 5))
    model.add(Activation("relu"))

    model.add(MaxPooling2D((2,2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
'''
'''
#cnn model identification

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='sigmoid'))
'''

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0005,decay=1e-6), metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=[X_test, Y_test])
# model.save('yawnModel.h5')

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])



'''
video_capture = cv2.VideoCapture()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if ret != 0 :
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
                )

    # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
im = cv2.imread('test.jpg')

model.predict(im, batch_size=1, verbose=1, steps=None)'''