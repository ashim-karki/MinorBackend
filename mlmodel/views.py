from rest_framework.response import Response
from django.http import JsonResponse
from rest_framework.decorators import api_view

from .serializers import ImageSerializer
from .models import ImageModel

import os
import tensorflow as tf
import cv2
import numpy as np
import base64
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import pandas as pd
from scipy.spatial import distance

# Load the trained model
model = tf.keras.models.load_model(
    "mlmodel/models/trial_model.h5"
)  # place model from drive

client_id = "cliend id here"
client_secret = "client secret here"
client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

df = pd.read_csv("mlmodel/songs_dataset/songs_dataset.csv", sep=",")


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
