import tensorflow as tf
# from keras.utils import np_utils
from keras.utils import Sequence
import matplotlib.pyplot as plt
import shutil
import os
import scipy
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import InceptionV3

original_dataset_dir_dogs = 'ImageRec/Dog'
original_dataset_dir_cats = 'ImageRec/Cat'
original_dataset_dir_elephants = 'ImageRec/elephant'

base_dir = '/Users/collinfox/PycharmProjects/tensorProject2/ImageRec/small'

train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    print('creating new train sudirectory for small dataset')
    os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    print('creating new validation su directory for small dataset')
    os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    print('creating new test subdirectory for small dataset')
    os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)

train_elephant_dir = os.path.join(train_dir, 'elephants')
if not os.path.exists(train_elephant_dir):
    os.mkdir(train_elephant_dir)

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validation_cats_dir):
    os.mkdir(validation_cats_dir)

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dogs_dir):
    os.mkdir(validation_dogs_dir)

# Directory with our elephant pictures
validation_elephant_dir = os.path.join(validation_dir, 'elephants')
if not os.path.exists(validation_elephant_dir):
    os.mkdir(validation_elephant_dir)

# Directory with our validation cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
if not os.path.exists(test_cats_dir):
    os.mkdir(test_cats_dir)

# Directory with our validation dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)

# Directory with elephant pictures
test_elephant_dir = os.path.join(test_dir, 'elephants')
if not os.path.exists(test_elephant_dir):
    os.mkdir(test_elephant_dir)

if len(os.listdir(train_cats_dir)) == 0:
    # Copy first 400 cat images to train_cats_dir
    fnames = ['{}.jpg'.format(i) for i in range(600)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_cats, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(validation_cats_dir)) == 0:
    # Copy next 200 cat images to validation_cats_dir
    fnames = ['{}.jpg'.format(i) for i in range(600, 800)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_cats, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(test_cats_dir)) == 0:
    # Copy next 400 cat images to test_cats_dir
    fnames = ['{}.jpg'.format(i) for i in range(800, 1200)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_cats, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(train_dogs_dir)) == 0:
    # Copy first 400 dog images to train_dogs_dir
    fnames = ['{}.jpg'.format(i) for i in range(600)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_dogs, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(validation_dogs_dir)) == 0:
    # Copy next 200 dog images to validation_dogs_dir
    fnames = ['{}.jpg'.format(i) for i in range(600, 800)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_dogs, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(test_dogs_dir)) == 0:
    # Copy next 400 dog images to test_dogs_dir
    fnames = ['{}.jpg'.format(i) for i in range(800, 1200)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_dogs, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(train_elephant_dir)) == 0:
    # Copy first 400 elephant images to train_elephant_dir
    fnames = ['{}.jpg'.format(i) for i in range(600)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_elephants, fname)
        dst = os.path.join(train_elephant_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(validation_elephant_dir)) == 0:
    # Copy next 200 elephant images to validation_elephant_dir
    fnames = ['{}.jpg'.format(i) for i in range(600, 800)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_elephants, fname)
        dst = os.path.join(validation_elephant_dir, fname)
        shutil.copyfile(src, dst)

if len(os.listdir(test_elephant_dir)) == 0:
    # Copy next 400 dog images to test_elephants_dir
    fnames = ['{}.jpg'.format(i) for i in range(800, 1200)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir_elephants, fname)
        dst = os.path.join(test_elephant_dir, fname)
        shutil.copyfile(src, dst)

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    # This is the target directory
    train_dir,
    # All images will be resized to 150x150
    target_size=(150, 150),
    batch_size=20,

    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

conv_base = InceptionV3(weights='imagenet',
                        include_top=False,
                        input_shape=(150, 150, 3))
print(conv_base.summary())


e_to_e_model = models.Sequential()
e_to_e_model.add(conv_base)
e_to_e_model.add(layers.BatchNormalization())
e_to_e_model.add(layers.GlobalAvgPool2D())


e_to_e_model.add(layers.Flatten())
e_to_e_model.add(layers.Dropout(.3))
e_to_e_model.add(layers.Dense(2048, activation='relu'))
e_to_e_model.add(layers.Dense(1024, activation='relu'))
e_to_e_model.add(layers.Dense(3, activation='softmax'))

conv_base.trainable = True

set_trainable = False

# Unfreezing some layers to train
for layer in conv_base.layers[:140]:
  layer.trainable=False

for layer in conv_base.layers[140:]:
  layer.trainable=True


e_to_e_model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                     metrics=['acc'])

e_to_e_history = e_to_e_model.fit(
    train_generator,
    steps_per_epoch=20,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=10,
    verbose=2)

e_acc = e_to_e_history.history['acc']
e_val_acc = e_to_e_history.history['val_acc']
e_loss = e_to_e_history.history['loss']
e_val_loss = e_to_e_history.history['val_loss']

e_epochs = range(len(e_acc))

plt.plot(e_epochs, e_acc, 'bo', label='Training acc')
plt.plot(e_epochs, e_val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(e_epochs, e_loss, 'bo', label='Training loss')
plt.plot(e_epochs, e_val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
plt.clf()


e_test_loss, e_test_acc = e_to_e_model.evaluate(test_generator, steps=20)
print('test acc:', e_test_acc)
