# Instantiate convolutional base
from keras_vggface.vggface import VGGFace
import os
from tqdm import tqdm
import shutil
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle

# Base variables
base_dir = '../'
train_dir = os.path.join(base_dir, 'femaleTrain')
validation_dir = os.path.join(base_dir, 'femaleValidation')
test_dir = os.path.join(base_dir, 'femaleTest')

# manually defined, number of samples we have for each category
train_size, validation_size, test_size = 6410, 2312, 2152

img_width, img_height = 224, 224  # Default input size for VGG16

conv_base = VGGFace(include_top=False, input_shape=(img_width, img_height, 3))
print(conv_base.summary())

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width,img_height),
                                            batch_size = batch_size,
                                            class_mode='sparse')
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in tqdm(generator):
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, train_size)

validation_features, validation_labels = extract_features(validation_dir, validation_size)

print(train_features)
print(train_labels)

with open("female_train_features.obj", "wb") as f:
    pickle.dump(train_features, f)

with open("female_train_labels.obj", "wb") as f:
    pickle.dump(train_labels, f)

with open("female_validation_features.obj", "wb") as f:
    pickle.dump(validation_features, f)

with open("female_validation_labels.obj", "wb") as f:
    pickle.dump(validation_labels, f)
