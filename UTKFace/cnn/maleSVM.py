import pickle
from sklearn import svm
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
import numpy as np
from os import path

# Define function to visualize predictions
def visualize_predictions(classifier):
    img_width, img_height = 224, 224  # Default input size for VGG16
    conv_base = VGGFace(include_top=False, input_shape=(img_width, img_height, 3))

    # Get picture
    base_dir = '../'
    img_path = path.join(base_dir, 'maleValidation/15-17', '17_0_4_20170103212532692.jpg.chip.jpg')
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
    img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application

    # Extract features
    features = conv_base.predict(img_tensor.reshape(1,img_width, img_height, 3))

    # Make prediction
    try:
        prediction = classifier.predict(features)
    except:
        prediction = classifier.predict(features.reshape(1, 7*7*512))

    # Show picture
    #plt.imshow(img_tensor)
    #plt.show()

    # Write prediction
    [prediction] = prediction
    labels = ['1-2', '10-14', '15-17', '18-21', '22-25', '26-30', '3-5', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '6-9', '61-65', '66-70']
    print(labels[int(prediction)])
    """
    if prediction < 0.5:
        print('Cat')
    else:
        print('Dog')
    """

with open('male_train_features.obj', 'rb') as f:
    train_features = pickle.load(f)
with open('male_train_labels.obj', 'rb') as f:
    train_labels = pickle.load(f)

with open('male_validation_features.obj', 'rb') as f:
    validation_features = pickle.load(f)
with open('male_validation_labels.obj', 'rb') as f:
    validation_labels = pickle.load(f)

# Concatenate training and validation sets
svm_features = np.concatenate((train_features, validation_features))
svm_labels = np.concatenate((train_labels, validation_labels))

X_train, y_train = svm_features.reshape(9414,7*7*512), svm_labels

if not path.exists('maleClf.obj'):
    print('clf does not exist yet. Computing...')

    param = [{
              "C": [0.01, 0.1, 1, 10, 100]
             }]

    svm = LinearSVC(penalty='l2', loss='squared_hinge')  # As in Tang (2013)
    clf = GridSearchCV(svm, param, cv=10)
    clf.fit(X_train, y_train)

    with open('maleClf.obj', 'wb') as f:
        pickle.dump(clf, f)
else:
    print('clf exists. Loading...')
    with open('maleClf.obj', 'rb') as f:
        clf = pickle.load(f)
# Save model
#model.save('ages_svm.h5')

#clf = svm.SVC(kernel='linear').fit(male_features, male_labels)

# prepare the image for VGG
"""
imgLocation = '../maleValidation/31-35/35_0_0_20170117122020502.jpg.chip.jpg'
img = cv2.imread(imgLocation, cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
img = img[np.newaxis, :, :, :]
# call feature extraction
clf.predict(img)
"""

visualize_predictions(clf)

# Evaluate model
"""
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

print("\nAccuracy score (mean):")
print(np.mean(cross_val_score(clf, X_train, y_train, cv=10)))
print("\nAccuracy score (standard deviation):")
print(np.std(cross_val_score(clf, X_train, y_train, cv=10)))
"""
