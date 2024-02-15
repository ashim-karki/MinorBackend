import cv2 
import numpy as np
# import matplotlib.pyplot as plt
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(imagePath, imageSize=48):
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        print("No face detected.")
        return None  # Return None to indicate no face detected

    # Assuming you want to work with the first detected face
    x, y, w, h = faces[0]
    crop_img = gray[y:y + h, x:x + w]
    cropped = cv2.resize(crop_img, (imageSize, imageSize))
    print(cropped.shape)
    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    
    # Return the cropped face image
    return cropped


#     plt.figure(figsize=(48,48))
#     plt.imshow(img_rgb)
#     plt.axis('off')
#     plt.show()
# detect_face('aiexpo.jpg')