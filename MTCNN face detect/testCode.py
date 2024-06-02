import numpy as np
import cv2 as cv
import face_recognition
import os

def extractFeatures(url, name):
    encodings = []
    names = []
    print(url)
    count = 0

    for imgUrl in os.listdir(url):
        print(imgUrl)
        pathImg = os.path.join(url, imgUrl)
        img = face_recognition.load_image_file(pathImg)

        en = face_recognition.face_encodings(img)
        if len(en) > 0:
            # print(len(en))
            encodings.append(en[0])
            names.append(name)
            count += 1

    print(f"Load success {count} images")

    return encodings, names

def loadDir(dirUrl):
    features = []
    labels = []

    for subUrl in os.listdir(dirUrl):
        feature, label = extractFeatures(os.path.join(dirUrl, subUrl), subUrl)
        featuresArr = []
        labelsArr = []
        featuresArr.append(np.asarray(feature))
        labelsArr.append(np.asarray(label))

        print(f"Label: {labelsArr}")
        features.append(featuresArr)
        labels.append(labelsArr)

    return features, labels

path = 'Train'
fea, lab = loadDir(path)

for index in range(len(fea)):
    np.save(f'Encoded_dataset\\Features\\feature {index}.npy', fea[index])
    np.save(f'Encoded_dataset\\Labels\\label {index}.npy', lab[index])