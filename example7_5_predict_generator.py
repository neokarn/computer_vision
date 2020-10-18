from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd


BATCH_SIZE = 5
IMAGE_SIZE = (256,256)

#Download dataset form https://drive.google.com/file/d/1jwa16s2nZIQywKMdRkpRvdDifxGDxC3I/view?usp=sharing
dataframe = pd.read_csv('rice/rice_weights.csv', delimiter=',', header=0)

datagen_noaug = ImageDataGenerator(rescale=1./255)

test_generator = datagen_noaug.flow_from_dataframe(
    dataframe=dataframe.loc[180:199],
    directory='rice/images',
    x_col='filename',
    y_col='norm_weight',
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='other')

model = load_model('rice_best.h5')
score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator))
print('score (mse, mae):\n',score)


test_generator.reset()
predict = model.predict_generator(
    test_generator,
    steps=len(test_generator),
    workers = 1,
    use_multiprocessing=False)
print('prediction:\n',predict)
