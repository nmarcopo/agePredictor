import pickle
from sklearn import svm
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import numpy as np

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

param = [{
          "C": [0.01, 0.1, 1, 10, 100]
         }]
 
svm = LinearSVC(penalty='l2', loss='squared_hinge')  # As in Tang (2013)
clf = GridSearchCV(svm, param, cv=10)
clf.fit(X_train, y_train)

with open('maleClf.obj', 'wb') as f:
    pickle.dump(clf, f)
# Save model
#model.save('ages_svm.h5')

#clf = svm.SVC(kernel='linear').fit(male_features, male_labels)

# prepare the image for VGG
imgLocation = '../maleValidation/31-35/35_0_0_20170117122020502.jpg.chip.jpg'
img = cv2.imread(imgLocation, cv2.CV_LOAD_IMAGE_COLOR)
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
img = img[np.newaxis, :, :, :]
# call feature extraction
clf.predict(img)
