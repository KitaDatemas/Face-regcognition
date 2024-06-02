import os.path

import face_recognition
import numpy as np
import cv2 as cv
from facenet_pytorch import MTCNN
import torch
import os

featuresPath = 'Encoded_dataset\\Features'
labelsPath = 'Encoded_dataset\\Labels'
features = []
labels = []
no_encoded_faces = 0

idx = 0
for subDir in os.listdir(featuresPath):
    f = []
    fTemp = np.load(f'Encoded_dataset\\Features\\feature {idx}.npy')[0].tolist()
    for fIdx in range(len(fTemp)):
        f.append(np.asarray(fTemp[fIdx]))
    features.append(f)
    idx += 1

no_encoded_faces = idx
idx = 0
for subDir in os.listdir(labelsPath):
    labels.append(np.load(f'Encoded_dataset\\Labels\\label {idx}.npy')[0].tolist())
    idx += 1

print(features)
print(len(features))
print(labels)
print(len(labels))

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

mtcnn = MTCNN(thresholds=[0.6, 0.7, 0.7], keep_all=True, device=device)

# img = 'received_475554161262906.jpg'
#
# img = cv.imread(img)
#
# boxes, _, _ = mtcnn.detect(img, landmarks=True)
#
# if boxes is not None:
#     for box in boxes:
#         image = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
#         cv.imshow("Crop", image)
#         cv.waitKey(0)
#         img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#         name = "Unknown"
#         en = face_recognition.face_encodings(img, face_recognition.face_locations(img))
#         if len(en) > 0:
#             en = face_recognition.face_encodings(img, face_recognition.face_locations(img))[0]
#             matches = face_recognition.compare_faces(features, en)
#             face_distances = face_recognition.face_distance(features, en)
#             best_match_index = np.argmin(face_distances)
#             if matches[best_match_index] is True:
#                 name = labels[best_match_index]
#                 break
#         cv.imshow(name, image)
#         cv.waitKey(0)

capture = cv.VideoCapture(0)

name = "Unknown"
confidence = 0

while True:
    isSuccess, frame = capture.read()
    frame = cv.flip(frame, 1)

    if isSuccess:
        boxes, prob, points = mtcnn.detect(frame, landmarks=True)

        if boxes is not None:
            for (box, point) in zip(boxes, points):
                x1, y1, x2, y2 = box.tolist()

                image = frame[int(y1) - 20:int(y2) + 20, int(x1) - 20:int(x2) + 20]
                try:
                    img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                except:
                    continue
                if cv.waitKey(1) & 0xFF == ord('c'):
                    # cv.imshow("crop ", image)
                    # cv.waitKey(0)
                    en = face_recognition.face_encodings(img, face_recognition.face_locations(img))
                    name = "Unknown"
                    if len(en) > 0:
                        # print("Checking")
                        confidence = 0.0
                        for index in range(no_encoded_faces):
                            en = face_recognition.face_encodings(img, face_recognition.face_locations(img))[0]
                            matches = face_recognition.compare_faces(features[index], en, tolerance=0.45)
                            # print("Matches: ")
                            # print(matches)
                            # cv.imshow("face", image)
                            # cv.waitKey(0)
                            # print(f"Best match index: {best_match_index}")

                            if matches.count(True)/len(matches) > 0.75:
                                name = labels[index][matches.index(True)]
                                confidence = matches.count(True) / len(matches)
                                break
                            else:
                                confidence = 1 - matches.count(True)/len(matches)


                    else:
                        print("else")
                        confidence = 0

                    for p in point:
                        cv.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 0), 2)

                cv.putText(frame, name, (int(x1), int(y1) - 50), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2)
                cv.putText(frame, f"{confidence}", (int(x1) , int(y1) - 10),
                           fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2)
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        else:
            name = "Unknown"
            confidence = 0
        cv.imshow('Camera', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
