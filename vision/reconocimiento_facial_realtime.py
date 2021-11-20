from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
#from tensorflow import keras
import pandas as pd

# Deepface es un paquete de reconocimiento facial híbrido.
# source: https://github.com/serengil/deepface

metrica_similaridad = ["cosine", "euclidean", "euclidean_l2"]
detectores = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]
modelos = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
demografico = ["age", "gender", "race", "emotion"]
database = "photos/" 

### 5. Streaming y análisis en tiempo real 
# Procesamiento de videos en tiempo real. La función Stream accederá a nuestra cámara web y aplicará tanto el reconocimiento facial como el análisis de atributos faciales. 
# La función comienza a analizar un fotograma si puede enfocar un rostro secuencialmente 5 fotogramas. Luego, muestra los resultados 5 segundos.
# RetinaFace y MTCNN parecen tener un rendimiento superior en las etapas de detección y alineación, pero son más lentos que otros. 
# Si la velocidad de su canalización es más importante, entonces debería usar opencv o ssd. Por otro lado, si considera la precisión, debe usar retinaface o mtcnn.

DeepFace.stream(db_path = "photos/",detector_backend = detectores[4], distance_metric = metrica_similaridad[2], model_name = modelos[0])

