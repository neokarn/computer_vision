from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

#Colab Version: https://colab.research.google.com/drive/1uhTC-qslMw2J8XBqAIZa9UWvKc9rEYk3?usp=sharing
#Download dataset from https://drive.google.com/open?id=1wWuxCQJEOQX980LuwSjTBM-EzbOJQtJy


BATCH_SIZE = 5
MAX_EPOCH = 30
IMAGE_SIZE = (256,256)
TRAIN_IM = 160
VALIDATE_IM = 15

model = Sequential()
model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                 input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(UpSampling2D(size=(2, 2)))
model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal'))

print(model.summary())

model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])

def myGenerator(type):
    datagen = ImageDataGenerator(rescale=1./255)

    input_generator = datagen.flow_from_directory(
        'textlocalize/'+type,
        class_mode=None,
        color_mode='rgb',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    expected_output_generator = datagen.flow_from_directory(
        'textlocalize/'+type,
        class_mode=None,
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    while True:
        in_batch = input_generator.next()
        out_batch = expected_output_generator.next()
        yield in_batch, out_batch

checkpoint = ModelCheckpoint('my_model.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

h = model.fit_generator(myGenerator('train'),
                        steps_per_epoch=TRAIN_IM/BATCH_SIZE,
                        epochs=MAX_EPOCH,
                        validation_data=myGenerator('validation'),
                        validation_steps=VALIDATE_IM/BATCH_SIZE,
                        callbacks=[checkpoint])

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.show()
