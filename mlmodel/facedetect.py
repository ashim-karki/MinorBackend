import cv2 
import numpy as np
# import matplotlib.pyplot as plt
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(imagePath, imageSize=224):
    img = cv2.imread(imagePath)
    face = face_detector.detectMultiScale(img, 1.1, 5, minSize=(40, 40))

    if len(face) > 0:
        x, y, w, h = face[0]
        crop_img = img[y:y+h, x:x+w]
        cropped = cv2.resize(crop_img, (imageSize, imageSize))
        img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        
        # plt.figure(figsize=(5, 5))
        # plt.imshow(img_rgb)
        # plt.axis('off')
        # plt.show()

        return img_rgb
    else:
        print("No face detected.")
        return None

#     plt.figure(figsize=(48,48))
#     plt.imshow(img_rgb)
#     plt.axis('off')
#     plt.show()
# detect_face('aiexpo.jpg')