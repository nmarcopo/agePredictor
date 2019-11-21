# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame
# ___________________________________________
# Andrey Kuehlkamp, Adam Czajka, November 2017

import cv2
import os
import sys
import numpy as np
from sklearn import svm
from keras.applications.vgg16 import VGG16
from keras.models import Model

# *** TASK 2:
# layer of the VGG features that will be used
cnn_codes = 'fc2'
clf = None

# an instance of VGG16: we need it to extract features (below)
model = VGG16(weights='imagenet')
# an alternative model, to extract features from the specified layer
# note that we could extract from any VGG16 layer, by name
features_model = Model(inputs=model.input, outputs=model.get_layer(cnn_codes).output)

# *** TASK 1 and TASK 3:
# we are going to use this list to restrict the objets our classifier will recognize
my_object_list = ['watch','dolphin','stop_sign','crab','camera','rooster','anchor','panda','scissors','stapler','headphone']

def classify_svm(img):
    features = extract_vgg_features(img)
    pred = clf.predict(features)

    # show classification result
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, '{}'.format(pred), (15, 25), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def extract_vgg_features(img):
    # prepare the image for VGG
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = img[np.newaxis, :, :, :]
    # call feature extraction
    return features_model.predict(img)


def camera_loop():
    print("Press <SPACE> to capture/classify an image, or <Esc> to exit.")
    cap = cv2.VideoCapture(0)
    while (True):
        _, frame = cap.read()

        action = cv2.waitKey(1)

        #img_to_show = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        #cv2.imshow('camera', img_to_show)

        if action == ord('q') or action == 27:
            break

            #if action == ord(' '):
            # svm object detection
        frame = classify_svm(frame)
        img_to_show = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
        cv2.imshow('SVM output:', img_to_show)

    cap.release()


if __name__ == '__main__':
    vggfile = 'vgg_features_{}.npz'.format(cnn_codes)

    # train the SVM to detect selected objects
    if os.path.exists(vggfile):
        # load pre-extracted features for all objects in Caltech 101
        print('Loading pre-extracted VGG features...')
        npzfile = np.load(vggfile)
        vgg_features = npzfile['vgg_features']
        labels = npzfile['labels']

        # filter out only the desired objects
        valid_indices = [n for n, l in enumerate(labels) if l in my_object_list]
        vgg_features = vgg_features[valid_indices]
        labels = labels[valid_indices]
    else:
        print("Pre-extracted features not found:", vggfile)
        sys.exit(0)

    # *** TASK 2:
    print("Training SVM ...")

    print(vgg_features.shape)
    print(labels.shape)
    

    #clf = svm.SVC(kernel='linear').fit(vgg_features, labels)
    #clf = svm.SVC(kernel='poly',degree=3).fit(vgg_features, labels)
    clf = svm.SVC(kernel='rbf',gamma='auto').fit(vgg_features, labels)

    camera_loop()
    cv2.destroyAllWindows()
