#Colab Version: https://colab.research.google.com/drive/1k3PclpSOtHmnlWnv7P6zAG-KZw6nxZV-?usp=sharing

from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

IM_SIZE = 224

#Create Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IM_SIZE, IM_SIZE, 3))
#base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IM_SIZE, IM_SIZE, 3))
#base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IM_SIZE, IM_SIZE, 3))

encoder_feature_map = base_model.output
avg_feature_map = GlobalAveragePooling2D()(encoder_feature_map)
dense = Dense(64, activation='relu')(avg_feature_map)
output = Dense(12, activation='softmax')(dense)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#Create generator (download dataset form https://drive.google.com/drive/folders/1jiwbiSbEMQkVGg3Oq2TZI2M9CPm0cJwC?usp=sharing)
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    '/animalfaces/train',
    shuffle=True,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=50,
    color_mode = 'rgb',
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    '/animalfaces/validation',
    shuffle=False,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=50,
    color_mode='rgb',
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    '/animalfaces/test',
    shuffle=False,
    target_size=(IM_SIZE,IM_SIZE),
    batch_size=50,
    color_mode='rgb',
    class_mode='categorical')


#Train Model
checkpoint = ModelCheckpoint('animalfaces2.keras', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')

h = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint])

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['train', 'val'])


#Test Model

model = load_model('animalfaces2.keras')
score = model.evaluate(
    test_generator,
    steps=len(test_generator))
print('score (cross_entropy, accuracy):\n',score)


test_generator.reset()
predict = model.predict(
    test_generator,
    steps=len(test_generator))
print('confidence:\n', predict)

predict_class_idx = np.argmax(predict,axis = -1)
print('predicted class index:\n', predict_class_idx)

mapping = dict((v,k) for k,v in test_generator.class_indices.items())
predict_class_name = [mapping[x] for x in predict_class_idx]
print('predicted class name:\n', predict_class_name)

cm = confusion_matrix(test_generator.classes, np.argmax(predict,axis = -1))
print("Confusion Matrix:\n",cm)

plt.show()

