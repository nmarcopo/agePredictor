# Instantiate convolutional base
from keras_vggface.vggface import VGGFace
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle
from tqdm import tqdm

# Base variables
base_dir = '../'
train_dir = os.path.join(base_dir, 'maleTrain')
validation_dir = os.path.join(base_dir, 'maleValidation')
test_dir = os.path.join(base_dir, 'maleTest')

# manually defined, number of samples we have for each category
train_size, validation_size, test_size = 7064, 2350, 2371

img_width, img_height = 224, 224  # Default input size for VGG16

conv_base = VGGFace(include_top=False, input_shape=(img_width, img_height, 3), model='resnet50')
print(conv_base.summary())

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 2048))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width,img_height),
                                            batch_size = batch_size,
                                            class_mode='sparse')
    print(generator.class_indices)
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
print("saving features...")
with open("male_train_features_resnet.obj", "wb") as f:
    pickle.dump(train_features, f, protocol=4)

with open("male_train_labels_resnet.obj", "wb") as f:
    pickle.dump(train_labels, f, protocol=4)

with open("male_validation_features_resnet.obj", "wb") as f:
    pickle.dump(validation_features, f, protocol=4)

with open("male_validation_labels.obj_resnet", "wb") as f:
    pickle.dump(validation_labels, f, protocol=4)
