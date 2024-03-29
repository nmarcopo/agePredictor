import pickle
from sklearn import svm
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
import numpy as np
from os import path
from os import walk
from tqdm import tqdm
from multiprocessing import Pool

# Define function to visualize predictions
def get_prediction(classifier, img_path, debug=False):
    img_width, img_height = 224, 224  # Default input size for VGG16
    conv_base = VGGFace(include_top=False, input_shape=(img_width, img_height, 3), model='resnet50', pooling=None)
    
    conv_base.layers.pop()
    conv_base.outputs = [conv_base.layers[-1].output]
    conv_base.layers[-1].outbound_nodes = []

    # Get picture
    if not img_path:
        print("no image passed in, using default image from 26-30")
        truth = '26-30'
        base_dir = '../'
        img_path = path.join(base_dir, 'maleTest/26-30', '28_0_3_20170105175516710.jpg.chip.jpg')
    else:
        truth = img_path.split('/')[-2]
    if debug:
        print("image is:", img_path)
        print("truth for this image is:", truth)
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
    img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application

    # Extract features
    features = conv_base.predict(img_tensor.reshape(1,img_width, img_height, 3))

    # Make prediction
    try:
        prediction = classifier.predict(features)
    except:
        prediction = classifier.predict(features.reshape(1, 7*7*2048))

    # Show picture
    #plt.imshow(img_tensor)
    #plt.show()

    # Write prediction
    [prediction] = prediction
    labels = ['1-2', '10-14', '15-17', '18-21', '22-25', '26-30', '3-5', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '6-9', '61-65', '66-70']
    oneOff = {
            '1-2': ['1-2', '3-5'],
            '3-5': ['1-2', '3-5', '6-9'],
            '6-9': ['3-5', '6-9', '10-14'],
            '10-14': ['6-9', '10-14', '15-17'],
            '15-17': ['10-14', '15-17', '18-21'],
            '18-21': ['15-17', '18-21', '22-25'],
            '22-25': ['18-21', '22-25', '26-30'],
            '26-30': ['22-25', '26-30', '31-35'],
            '31-35': ['26-30', '31-35', '36-40'],
            '36-40': ['31-35', '36-40', '41-45'],
            '41-45': ['36-40', '41-45', '46-50'],
            '46-50': ['41-45', '46-50', '51-55'],
            '51-55': ['46-50', '51-55', '56-60'],
            '56-60': ['51-55', '56-60', '61-65'],
            '61-65': ['56-60', '61-65', '66-70'],
            '66-70': ['61-65', '66-70'],
            }
    predictionText = labels[int(prediction)]
    
    print("image:", img_path)
    print("prediction for image is:", predictionText)

    predictionList = []
    if predictionText != truth:
        if debug:
            print("PREDICTION WAS WRONG!!!!")
        predictionList.append(0)
    else:
        predictionList.append(1)

    if predictionText in oneOff[truth]:
        predictionList.append(1)
    else:
        predictionList.append(0)

    return predictionList

print("Loading training features")
with open('male_train_features_resnet.obj', 'rb') as f:
    train_features = pickle.load(f)
print("Loading training labels")
with open('male_train_labels_resnet.obj', 'rb') as f:
    train_labels = pickle.load(f)

print("Loading validation features")
with open('male_validation_features_resnet.obj', 'rb') as f:
    validation_features = pickle.load(f)
print("Loading validation labels")
with open('male_validation_labels_resnet.obj', 'rb') as f:
    validation_labels = pickle.load(f)

# Concatenate training and validation sets
print("creating training features and labels matricies")
svm_features = np.concatenate((train_features, validation_features))
svm_labels = np.concatenate((train_labels, validation_labels))

num_images_in_set = 9414
X_train, y_train = svm_features.reshape(num_images_in_set,7*7*2048), svm_labels

if not path.exists('maleClf_resnet.obj'):
    print('clf does not exist yet. Computing...')

    param = [{
              "C": [0.01, 0.1, 1, 10, 100]
             }]

    #svm = LinearSVC(penalty='l2', loss='squared_hinge')  # As in Tang (2013)
    svm = SVC(kernel='rbf', gamma='auto')
    clf = GridSearchCV(svm, param, cv=10)
    clf.fit(X_train, y_train)

    with open('maleClf_resnet.obj', 'wb') as f:
        pickle.dump(clf, f)
else:
    print('clf exists. Loading...')
    with open('maleClf_resnet.obj', 'rb') as f:
        clf = pickle.load(f)

def paralell_predictions(path):
    return get_prediction(clf, path[0] + '/' + path[1], False)

totalTested = 0
totalCorrect = 0

pool = Pool(processes=3)
for root, dirs, files in list(walk("../maleTest/"))[3:]:
    print(root)
    root = [root for _ in range(len(list(files)))]
    if len(root) == 0:
        continue
    print("there are", len(root), "files to go through.")
    #prediction_results = list(tqdm(pool.imap_unordered(paralell_predictions, zip(root, files))))
    currTested = 0
    currCorrect = 0
    oneOffCorrect = 0
    for response, oneOff in tqdm(pool.imap_unordered(paralell_predictions, zip(root, files))):
        currTested += 1
        currCorrect += response
        oneOffCorrect += oneOff
        print("current progress for", root[0],":", currTested, "/", len(root),"tested,", currCorrect, "correct,", float(currCorrect/currTested), "oneOff:", oneOffCorrect, float(oneOffCorrect / currTested))
    print("final progress for", root[0],":", currTested, "tested,", currCorrect, "correct,", float(currCorrect/currTested), "oneOff:", oneOffCorrect, float(oneOffCorrect / currTested))

"""
# Evaluate model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

print("\nAccuracy score (mean):")
print(np.mean(cross_val_score(clf, X_train, y_train, cv=10)))
#print("\nAccuracy score (standard deviation):")
#print(np.std(cross_val_score(clf, X_train, y_train, cv=10)))
"""
