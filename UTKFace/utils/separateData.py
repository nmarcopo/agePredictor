import random
import os
from shutil import copyfile

# THIS SCRIPT SHOULD BE IN ./UTILS/

femaleRanges = os.listdir("../female/")
maleRanges = os.listdir("../male/")

try:
    os.mkdir("../femaleTrain")
    os.mkdir("../femaleValidation")
    os.mkdir("../femaleTest")
except:
    print("female folders already created")

try:
    os.mkdir("../maleTrain")
    os.mkdir("../maleValidation")
    os.mkdir("../maleTest")
except:
    print("male folders already created!")

dataTypes = ["Train/", "Validation/", "Test/"]

for ageRange in femaleRanges:
    for data in dataTypes:
        ageRangePath = "../female" + data + ageRange
        try:
            os.mkdir(ageRangePath)
        except:
            print(ageRangePath, "already created")

for ageRange in maleRanges:
    for data in dataTypes:
        ageRangePath = "../male" + data + ageRange
        try:
            os.mkdir(ageRangePath)
        except:
            print(ageRangePath, "already created")

trainRatio = .6
validationRatio = .2
testRatio = .2

print("copying female files...")
for ageRange in femaleRanges:
    directory = "../female/" + ageRange
    files = os.listdir(directory)
    random.shuffle(files)
    trainLen = int(len(files) * trainRatio)
    validationLen = int(len(files) * validationRatio)
    
    # move training files
    for f in files[:trainLen]:
        copyfile(directory + "/" + f, "../femaleTrain/" + ageRange + '/' + f)

    # move validation files
    for f in files[trainLen:trainLen + validationLen]:
        copyfile(directory + "/" + f, "../femaleValidation/" + ageRange + '/' + f)

    # move training files
    for f in files[trainLen + validationLen:]:
        copyfile(directory + "/" + f, "../femaleTest/" + ageRange + '/' + f)


print("copying male files...")
for ageRange in maleRanges:
    directory = "../male/" + ageRange
    files = os.listdir(directory)
    random.shuffle(files)
    trainLen = int(len(files) * trainRatio)
    validationLen = int(len(files) * validationRatio)

    # move training files
    for f in files[:trainLen]:
        copyfile(directory + "/" + f, "../maleTrain/" + ageRange + '/' + f)

    # move validation files
    for f in files[trainLen:trainLen + validationLen]:
        copyfile(directory + "/" + f, "../maleValidation/" + ageRange + '/' + f)

    # move training files
    for f in files[trainLen + validationLen:]:
        copyfile(directory + "/" + f, "../maleTest/" + ageRange + '/' + f)
