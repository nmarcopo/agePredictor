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
from os import walk
from tqdm import tqdm
from multiprocessing import Pool

# Define function to visualize predictions
def get_prediction(classifier, img_path, debug=False):
    img_width, img_height = 224, 224  # Default input size for VGG16
    conv_base = VGGFace(include_top=False, input_shape=(img_width, img_height, 3))
 #   print(conv_base.summary())
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
        prediction = classifier.predict(features.reshape(1, 7*7*512))

    # Show picture
    #plt.imshow(img_tensor)
    #plt.show()

    # Write prediction
    [prediction] = prediction
    labels = ['1-2', '14-16', '17-20', '21-25', '26-30', '3-5', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '6-8', '61-70', '9-13']
    oneOff= {
        '1-2': ['1-2','3-5'],
        '3-5': ['1-2','3-5','6-8'],
        '6-8': ['3-5','6-8','9-13'],
        '9-13':['6-8','9-13','14-16'],
        '14-16':['9-13','14-16','17-20'],
        '17-20':['14-16','17-20','21-25'],
        '21-25':['17-20','21-25','26-30'],
        '26-30':['21-25','26-30','31-35'],
        '31-35':['26-30','31-35','36-40'],
        '36-40':['31-35','36-40','41-45'],
        '41-45':['36-40','41-45','46-50'],
        '46-50':['41-45','46-50','51-55'],
        '51-55':['46-50','51-55','56-60'],
        '56-60':['51-55','56-60','61-70'],
        '61-70':['56-60','61-70']
    }
    predictionList= []
    predictionText = labels[int(prediction)]
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
with open('female_train_features.obj', 'rb') as f:
    train_features = pickle.load(f)
print("Loading training labels")
with open('female_train_labels.obj', 'rb') as f:
    train_labels = pickle.load(f)

print("Loading validation features")
with open('female_validation_features.obj', 'rb') as f:
    validation_features = pickle.load(f)
print("Loading validation labels")
with open('female_validation_labels.obj', 'rb') as f:
    validation_labels = pickle.load(f)

# Concatenate training and validation sets
print("creating training features and labels matricies")
svm_features = np.concatenate((train_features, validation_features))
svm_labels = np.concatenate((train_labels, validation_labels))

num_images_in_set = 8542
X_train, y_train = svm_features.reshape(num_images_in_set,7*7*512), svm_labels

if not path.exists('femaleClfRbf.obj'):
    print('clf does not exist yet. Computing...')

    param = [{
              "C": [0.01, 0.1, 1, 10, 100]
             }]

    svm = svm.SVC(gamma='auto',kernel='rbf')  # As in Tang (2013)
    clf = GridSearchCV(svm, param, cv=10)
    clf.fit(X_train, y_train)

    with open('femaleClfRbf.obj', 'wb') as f:
        pickle.dump(clf, f)
else:
    print('clf exists. Loading...')
    with open('femaleClfRbf.obj', 'rb') as f:
        clf = pickle.load(f)

def paralell_predictions(path):
    return get_prediction(clf, path[0] + '/' + path[1], False)

totalTested = 0
totalCorrect = 0
pool = Pool(processes=7)
for root, dirs, files in list(walk("../femaleTest/")):
    if len(files) == 0:
        continue
    print(root)
    root = [root for _ in range(len(list(files)))]
    print("there are", len(root), "files to go through.")
    #prediction_results = list(tqdm(pool.imap_unordered(paralell_predictions, zip(root, files))))
    currTested = 0
    currCorrect = 0
    oneOffCorrect=0
    for response, oneOff in tqdm(pool.imap_unordered(paralell_predictions, zip(root, files))):
        currTested += 1
        currCorrect += response
        oneOffCorrect += oneOff
        print("current progress for", root[0],":", currTested, "/", len(root),"tested,", currCorrect, "correct,", float(currCorrect/currTested),"  one off: ",float(oneOffCorrect)," percent ",float(oneOffCorrect/currTested))
        print("final progress for", root[0],":", currTested, "tested,", currCorrect, "correct,", float(currCorrect/currTested),"oneoff correct:",oneOffCorrect,float(oneOffCorrect/currTested))
    #print("accuracy for", root, "is", float(sum(prediction_results) / len(prediction_results)))
    #print("there are", len(prediction_results), "files, and", sum(prediction_results), "were correct.")
    """
    dirTested = 0
    dirCorrect = 0
    for f in tqdm(files):
        predicted = get_prediction(clf, root + '/' + f, False)
        totalCorrect += predicted
        totalTested += 1

        dirCorrect += predicted
        dirTested += 1
    if dirTested:
        print("accuracy for", root, "is", float(dirCorrect / dirTested))
    """
print("accuracy for all folders is", float(totalCorrect / totalTested))
get_prediction(clf, None)

# Evaluate model
"""
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

print("\nAccuracy score (mean):")
print(np.mean(cross_val_score(clf, X_train, y_train, cv=10)))
print("\nAccuracy score (standard deviation):")
print(np.std(cross_val_score(clf, X_train, y_train, cv=10)))
"""
