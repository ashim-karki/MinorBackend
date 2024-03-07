from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework.decorators import api_view

from .serializers import ImageSerializer
from .models import ImageModel

import os , ast
import tensorflow as tf
import cv2
import numpy as np
import base64
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import pandas as pd
from scipy.spatial import distance
from .facedetect import detect_face


# Load the trained model
#model = tf.keras.models.load_model(
#    "mlmodel/models/trial_model.h5"
#)  # place model from drive
#model1 = tf.keras.models.load_model(
#    "mlmodel/models/emotion_classifier.h5"
#)
model2 = tf.keras.models.load_model("mlmodel/models/affectnet_CNN_VGG_FIVEEMO_FINE_FINAL.h5")
client_id = "cliend id here"
client_secret = "client secret here"
client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

df = pd.read_csv("mlmodel/songs_dataset/songs_dataset.csv", sep=",")
df1 = pd.read_csv("mlmodel/songs_dataset/temp.csv", sep=",")

# emotion_label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
emotion_label_dict = {
    0:'neutral',
    1:'happiness',
    2:'sadness',
    3:'surprise',
    4:'fear',
    5:'disgust',
    6: 'anger',
    7:'contempt',
   }
# Create your views here.
@api_view(["POST"])
def UseMlModel(request):
    if request.method == "POST":
        existing_check = ImageModel.objects.filter(tempid=1)
        if not existing_check:
            serialized_image = ImageSerializer(data=request.data)
        else:
            existing_image = ImageModel.objects.get(tempid=1)
            serialized_image = ImageSerializer(existing_image, data=request.data)

        if serialized_image.is_valid():
            serialized_image.save()

            image_name = request.data["image"].name
            image_path = "mlmodel/images/" + image_name

            with open(image_path, "rb") as image_file:
                base64_bytes = base64.b64encode(image_file.read())
                base64_string = base64_bytes.decode()

            image = cv2.imread(image_path)
            image = cv2.resize(image, (150, 150))
            image = np.expand_dims(image, axis=0)
            image = image / 255.0

        os.remove(image_path)

        prediction = model.predict(image)
        prediction = prediction.flatten()  # convert 2d to 1d array
        serializable_prediction = prediction.tolist()
        serializable_prediction.append(0.9446094)  # test_value

        # USING EUCLIDEAN DISTANCE
        Song_IDs = []
        Song_Names = []
        euclidean_distance = 0
        number_of_songs = len(df)
        for i in range(number_of_songs):
            case_values = [df["Valence"].values[i], df["Energy"].values[i]]
            euclidean_distance = distance.euclidean(
                serializable_prediction, case_values
            )
            if euclidean_distance < 0.3:
                Song_IDs.append(df["Song ID"].values[i])
                Song_Names.append(df["Song Name"].values[i])

        # def recommend(test_values, songs_data, n_recs):
        #     songs_data["mood_vectors"] = songs_data[
        #         ["Valence", "Energy"]
        #     ].values.tolist()
        #     songs_data.drop(axis=1, columns=["Valence", "Energy"])
        #     songs_data["distance"] = songs_data["mood_vectors"].apply(
        #         lambda x: np.linalg.norm((test_values) - np.array(x))
        #     )
        #     songs_data_sorted = songs_data.sort_values(by="distance", ascending=True)
        #     return songs_data_sorted.iloc[:n_recs]

        # recommend_df = recommend(
        #     test_values=serializable_prediction, songs_data=df, n_recs=10
        # )
        # print(recommend_df["Song Name"])

        return JsonResponse(
            {
                "Prediction": serializable_prediction,
                "Song Names": Song_Names,
                "Song ID": Song_IDs,
                # "Image" :base64_string,
            }
        )


@api_view(["POST"])
def UseMlModelTemp(request):
    if request.method == "POST":
        existing_check = ImageModel.objects.filter(tempid=1)
        if not existing_check:
            serialized_image = ImageSerializer(data=request.data)
        else:
            existing_image = ImageModel.objects.get(tempid=1)
            serialized_image = ImageSerializer(existing_image, data=request.data)

        if serialized_image.is_valid():
            serialized_image.save()

            image_name = request.data["image"].name
            image_path = "mlmodel/images/" + image_name

            with open(image_path, "rb") as image_file:
                base64_bytes = base64.b64encode(image_file.read())
                base64_string = base64_bytes.decode()

            image = cv2.imread(image_path)
            image = cv2.resize(image, (150, 150))
            image = np.expand_dims(image, axis=0)
            image = image / 255.0

            os.remove(image_path)

        prediction = model.predict(image)
        prediction = prediction.flatten()  # convert 2d to 1d array
        serializable_prediction = prediction.tolist()
        serializable_prediction.append(0.6446094)  # test_value

        # USING EUCLIDEAN DISTANCE
        music_data_list = []
        number_of_songs = len(df)
        euclidean_distance = 0
        
        for i in range(number_of_songs):
            case_values = [df["valence"].values[i], df["energy"].values[i]]
            euclidean_distance = distance.euclidean(
                serializable_prediction, case_values
            )
            print(euclidean_distance)
            if euclidean_distance < 0.9:
                music_data = {
                    "Song Name": df["name"].values[i],
                    "Song ID": df["id"].values[i],
                    "Preview Url": df["preview_url"].values[i],
                }
                music_data_list.append(music_data)

        return JsonResponse({"Predictions": serializable_prediction, "Music Data": music_data_list})


@api_view(["POST"])
def DetectEmotion(request):
    if request.method == "POST":
        existing_check = ImageModel.objects.filter(tempid=1)
        if not existing_check:
            serialized_image = ImageSerializer(data=request.data)
        else:
            existing_image = ImageModel.objects.get(tempid=1)
            serialized_image = ImageSerializer(existing_image, data=request.data)

        if serialized_image.is_valid():
            serialized_image.save()

            image_name = request.data["image"].name
            image_path = "mlmodel/images/" + image_name

            with open(image_path, "rb") as image_file:
                base64_bytes = base64.b64encode(image_file.read())
                base64_string = base64_bytes.decode()
            # reuturn emotion Class
        


        
        emotionClass,emotionWeight = classify(imagePath=image_path)
        emotionName = emotion_label_dict[emotionClass]
        

        os.remove(image_path)

        predicted_values = np.array(mapEmotionToValAro(emotionClass,emotionWeight))
        recommendMusicDF = recommend(predicted_values,df1,9)
        recommended_songs_dict = recommendMusicDF.to_dict(orient='records')

    return JsonResponse({"Music Data":recommended_songs_dict,"Emotion": emotionName})

        
def recommend(test_values, songs_data, n_recs):
    songs_data["distance"] = songs_data["mood_vectors"].apply(
        lambda x: np.linalg.norm(test_values - np.array(ast.literal_eval(x)))
    )
    songs_data_sorted = songs_data.sort_values(by="distance", ascending=True)
    return songs_data_sorted.iloc[:n_recs][["name", "id", "preview_url"]].rename(columns={"name": "Song Name", "id": "Song ID", "preview_url": "Preview Url"})

        

def load_image(imagePath):
    img = detect_face(imagePath)
    img = np.array(img)
    return img 

def classify(imagePath):
    img = load_image(imagePath)
    img = np.expand_dims(img,axis = 0) #makes image shape (1,48,48)
    
    result = model2.predict(img)
    result = tf.nn.softmax(result)
    weight = np.max(result,axis = 1)
    result = np.argmax(result,axis = 1)
    
    # result = list(result[0])
    # img_index = result.index(max(result))
    
    # print(result)
    img_index  = result[0]
    # result = list(result[0])
    return img_index,weight

def mapEmotionToValAro(index,weight):
    if index == 0:
        return weight * [0.5, 0.5]  # neutral
    elif index == 1:
        return weight * [0.8, 0.8]  # happiness
    elif index == 2:
        return weight * [0.2, 0.2]  # sadness
    elif index == 3:
        return weight * [0.7, 0.8]  # surprise
    elif index == 4:
        return weight * [0.2, 0.7]  # fear
    elif index == 5:
        return weight * [0.2, 0.2]  # disgust
    elif index == 6:
        return [0.2, 0.8]  # anger
    elif index == 7:
        return [0.4, 0.4]  # contempt
    else:
        return None  # handle invalid index
