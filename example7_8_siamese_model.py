#Colab Version: https://colab.research.google.com/drive/1v9rXgul1Lg0imTFxFT7wlBJTAI6OzKRU?usp=sharing

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Concatenate, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import pandas as pd


BATCH_SIZE = 100
MAX_EPOCH = 20
IMAGE_SIZE = (128,128)
TRAIN_IM_PAIR = 2000
VALIDATE_IM_PAIR  = 200


#Create Model
input_encoder = Input(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
conv1 = Conv2D(32, (3, 3), activation='relu')(input_encoder)
pool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D((2, 2))(conv2)
conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
pool3 = MaxPooling2D((2, 2))(conv3)
conv4 = Conv2D(256, (3, 3), activation='relu')(pool3)
pool4 = MaxPooling2D((2, 2))(conv4)
conv5 = Conv2D(512, (3, 3), activation='relu')(pool4)
pool5 = MaxPooling2D((2, 2))(conv5)
flat = Flatten()(pool5)
encoder = Model(inputs=input_encoder, outputs=flat)

input_1 = Input(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
feature_1 = encoder(input_1)

input_2 = Input(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
feature_2 = encoder(input_2)

concat = Concatenate()([feature_1,feature_2])

dense1 = Dense(128, activation='relu')(concat)
output = Dense(1, activation='sigmoid')(dense1)

siamese_model = Model(inputs=[input_1, input_2], outputs=output)

siamese_model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])

plot_model(siamese_model, show_shapes=True)


#Create Generator (download dataset from https://drive.google.com/drive/folders/1jiwbiSbEMQkVGg3Oq2TZI2M9CPm0cJwC?usp=drive_link)

def myGenerator(type):
    dataframe = pd.read_csv('/animalfaces/'+type+'/image_pair_classify.csv', delimiter=',', header=0)

    datagen = ImageDataGenerator(rescale=1./255)

    input_generator_1 = datagen.flow_from_dataframe(dataframe = dataframe,
        directory='/animalfaces/'+type+'/',
        x_col='filename_1',
        y_col='class',
        class_mode='other',
        color_mode='rgb',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    input_generator_2 = datagen.flow_from_dataframe(dataframe = dataframe,
        directory='/animalfaces/'+type+'/',
        x_col='filename_2',
        y_col='class',
        class_mode='other',
        color_mode='rgb',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed = 1)

    while True:
        in_batch_1 = next(input_generator_1)
        in_batch_2 = next(input_generator_2)
        yield (in_batch_1[0], in_batch_2[0]), in_batch_1[1]

#Train Model
checkpoint = ModelCheckpoint('image_pair_classify.keras', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

h = siamese_model.fit(myGenerator('train'),
              steps_per_epoch=int(TRAIN_IM_PAIR/BATCH_SIZE),
              epochs=MAX_EPOCH,
              validation_data=myGenerator('validation'),
              validation_steps=int(VALIDATE_IM_PAIR/BATCH_SIZE),
              callbacks=[checkpoint])


