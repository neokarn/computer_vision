#Colab Version: https://colab.research.google.com/drive/1uhTC-qslMw2J8XBqAIZa9UWvKc9rEYk3?usp=sharing

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D , UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

BATCH_SIZE = 5
MAX_EPOCH = 10
IMAGE_SIZE = (256,256)
TRAIN_IM = 160
VALIDATE_IM = 15

model = Sequential()
model.add(Input(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3)))
model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
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


#Create generator (download dataset form https://drive.google.com/drive/folders/1etiED1j8f65YJktni36t7Wdj6M0lRZPy?usp=sharing)

def myGenerator(type):
    datagen = ImageDataGenerator(rescale=1./255)

    input_generator = datagen.flow_from_directory(
        '/textlocalize/'+type,
        classes = ['Input'],
        class_mode=None,
        color_mode='rgb',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    expected_output_generator = datagen.flow_from_directory(
        '/textlocalize/'+type,
        classes = ['Output'],
        class_mode=None,
        color_mode='grayscale',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    while True:
        in_batch = next(input_generator)
        out_batch = next(expected_output_generator)
        yield in_batch, out_batch


#Train Model
checkpoint = ModelCheckpoint('text_localize_model.keras', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')


h = model.fit(myGenerator('train'),
              steps_per_epoch=int(TRAIN_IM/BATCH_SIZE),
              epochs=MAX_EPOCH,
              validation_data=myGenerator('validation'),
              validation_steps=int(VALIDATE_IM/BATCH_SIZE),
              callbacks=[checkpoint])

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.show()




