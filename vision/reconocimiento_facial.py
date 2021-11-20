from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
#from tensorflow import keras
import pandas as pd
import json

# Deepface es un paquete de reconocimiento facial híbrido.
# source: https://github.com/serengil/deepface

# Papers: 
# On Visual BMI Analysis from Facial Images, https://pdf.sciencedirectassets.com/271526/1-s2.0-S0262885619X00086/1-s2.0-S0262885619301027/am.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIHbUEwuW%2BZg1keaDaAvC7IfEgE17kobx7%2Fn%2FqXx2ZWXMAiEAzO0soRiIgK3QM%2B9KcXV0d4mEEGl8hO1X%2F8bYIV3XGcwqgwQIr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwwNTkwMDM1NDY4NjUiDIcmQoqsJ6G%2BNcaXwyrXA3rMxsdoPC4UdPAnqNFuhpKIWiYPr0Z6nIjugFplAt8TkvAb0Etnte0gqU79pifxySk2YXPi%2FHCT8dx6oT6K5PpkXVRXort1%2Fdu7fera3QmtZbjSO%2BupIbQfU8Il23C70ZTzDSOOoQc4B585ZhJRBig0VgDk3JICTZjrIm%2FDFvOHyecpTovJUVAkiOxREX4fRjBxsUZLCtH6GUF%2BqdLr0uitiCr1VSXxA55zRy%2FNcvzSxjzXoz6WxwV4zGRfIiq1x4P6H86rVazseZNgFjvNxXctezIuYHL33slvTs0pxFoUlZjcZipdVekMfvJKtmTIexwszsglSMDmjRtRmWbuRjzgscPJlglJFYpciUCD3vIyfs8J3aoGpv0IgrNveTtS46XXw3S6sW43G%2FugEzN5jGOcbuOMVc8vqsEfiBUYwSEPyrisCCAizvff6KTMadK1qJjMSfe4JMNzQY29a3ddYuFm1g7CGPW%2FUaJOz400fgkfghHMcolBr5E5ULfPZyad5r7uaCJFmMMKCSA75ehaKWb56lKAlHvV5lF2oaEFTZi%2Fry6fMcraUo6vjr1QCCoOtvoqZq51MoI0mc5%2FqJYL7azkYtjUB20z68OeswjErPtbnj7xW6HApjDi6dWMBjqlAZ8FEWN8R5U8jdq1eDfGijhMcbCAEXChZboTJmIT5VbpsoALfAZ7Nc8lEGSt%2Bi2SOtdWC79Y6wItxW11zTr76PEGNben80FQyJtj7N3wmjsiU6uJIMmLXzOdI6QSc0i7ZdNWbSX2NIH0VFLSnf95q%2BGoK%2F%2FNcaP1nEJEmkNzoOSOn3fwBNgkwo5u8bR7OzwOUbUlXhZKZ806DHGNUDJ%2Bbo1MyHkQKg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20211117T230206Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQBLDAOL7%2F20211117%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=fd2c0f165acb2a0903278e96627867fb30e5a161cfaac579864c1afeaa3a9139&hash=b087fc6bd5c833bf96cf23240253e5e9a88a23af5389e468e8beb612ae36d91a&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0262885619301027&tid=pdf-82fef535-d5e8-48fd-98bb-8b40b4e138d8&sid=a10222ae4cc087477f2afbb5d3312d69facdgxrqa&type=client
# representations_vgg_face.pkl : La estrcura de datos de imagenes
# *.pk: https://newbedev.com/how-to-put-my-dataset-in-a-pkl-file-in-the-exact-format-and-data-structure-used-in-mnist-pkl-gz

# 1. Modelos de reconocimiento facial
# Actualmente incluye los modelos de reconocimiento facial de última generación: VGG-Face, Google FaceNet, OpenFace, Facebook DeepFace, DeepID, ArcFace y Dlib. 
# La configuración predeterminada verifica las caras con el modelo VGG-Face. 
# Puede configurar el modelo base durante la verificación como se muestra a continuación.
# FaceNet, VGG-Face, ArcFace y Dlib tienen un rendimiento superior al de OpenFace, DeepFace y DeepID según los experimentos. 
# Como apoyo, FaceNet obtuvo un 99,65%; ArcFace obtuvo el 99,40%; Dlib obtuvo el 99,38%; 
# VGG-Face obtuvo un 98,78%; OpenFace obtuvo puntuaciones de precisión del 93,80% en el conjunto de datos de LFW, 
# mientras que los seres humanos podrían tener solo el 97,53%.

# VGG 
# Son las siglas de Visual Geometry Group. La red neuronal VGG (VGGNet) es uno de los tipos de modelos de reconocimiento de imágenes más utilizados que se basa en redes neuronales 
# convolucionales profundas. La arquitectura VGG se hizo famosa por lograr los mejores resultados en el desafío ImageNet. El modelo está diseñado por investigadores de la Universidad de Oxford.
#Si bien el VGG-Face tiene la misma estructura que el modelo VGG normal, está sintonizado con imágenes faciales. El modelo de reconocimiento facial de VGG alcanza un 97,78% 
# de precisión en el popular conjunto de datos de Caras etiquetadas en la naturaleza (LFW).

# Google FaceNet
# Este modelo está desarrollado por los investigadores de Google. FaceNet se considera un modelo de vanguardia para la detección y el reconocimiento de rostros con aprendizaje profundo. FaceNet se puede utilizar para el reconocimiento facial, la verificación y la agrupación 
# (la agrupación de caras se utiliza para agrupar fotos de personas con la misma identidad).
# El principal beneficio de FaceNet es su alta eficiencia y rendimiento, se informa que alcanza un 99,63% de precisión en el conjunto de datos LFW y un 95,12% 
# en Youtube Faces DB, mientras que utiliza solo 128 bytes por cara.

# Openface
# Este modelo de reconocimiento facial está construido por investigadores de la Universidad Carnegie Mellon. Por lo tanto, OpenFace está fuertemente inspirado en el proyecto FaceNet, 
# pero este es más liviano y su tipo de licencia es más flexible. OpenFace logra una precisión del 93,80% en el conjunto de datos de LFW.
# Cómo usar OpenFace: al igual que con los modelos anteriores, puede usar el modelo OpenFace AI utilizando la biblioteca DeepFace.
# Cómo usar FaceNet: Probablemente la forma más fácil de usar Google FaceNet es con la biblioteca DeepFace, que puede instalar y establecer un argumento en las funciones de DeepFace

# Deepface
# Este modelo de reconocimiento facial fue desarrollado por investigadores de Facebook. El algoritmo de Facebook DeepFace se entrenó en un conjunto de datos etiquetados de cuatro millones de rostros pertenecientes a más de 4000 personas, que era el conjunto de datos faciales más grande en el momento del lanzamiento. El enfoque se basa en una red neuronal profunda con nueve capas.
# El modelo de Facebook alcanza una precisión del 97,35% (+/- 0,25%) en el punto de referencia del conjunto de datos de LFW. Los investigadores afirman que el algoritmo DeepFace de Facebook cerrará la brecha con el rendimiento a nivel humano (97,53%) en el mismo conjunto de datos. Esto indica que DeepFace a veces tiene más éxito que los seres humanos al realizar tareas de reconocimiento facial.
# Cómo usar Facebook DeepFace: una manera fácil de usar el algoritmo de reconocimiento facial de Facebook es usando la biblioteca DeepFace de nombre similar que contiene el modelo de Facebook.

# DeepID

# El algoritmo de verificación facial DeepID realiza un reconocimiento facial basado en el aprendizaje profundo. Fue uno de los primeros modelos que utilizó redes neuronales convolucionales y 
# logró un rendimiento mejor que el humano en tareas de reconocimiento facial. 
# Deep-ID fue introducido por investigadores de la Universidad China de Hong Kong.
# Los sistemas basados ​​en el reconocimiento facial DeepID fueron algunos de los primeros en superar el desempeño humano en la tarea. Por ejemplo, 
# DeepID2 alcanzó el 99,15% en el conjunto de datos de Caras etiquetadas en la naturaleza (LFW).

# Dlib
# El modelo de reconocimiento facial Dlib se autodenomina "la API de reconocimiento facial más simple del mundo para Python". El modelo de aprendizaje automático se utiliza para reconocer y manipular caras desde Python o desde la línea de comandos. Si bien la biblioteca dlib está escrita originalmente en C ++, tiene enlaces de Python fáciles de usar.
# Curiosamente, el modelo Dlib no fue diseñado por un grupo de investigación. Es presentado por Davis E. King, el desarrollador principal de la biblioteca de procesamiento de imágenes Dlib.
# La herramienta de reconocimiento facial de Dlib mapea una imagen de un rostro humano en un espacio vectorial de 128 dimensiones, donde las imágenes de la misma persona están cerca una de la otra y las imágenes de diferentes personas están muy alejadas. Por lo tanto, dlib realiza el reconocimiento facial mapeando caras al espacio 128d y luego verificando si su distancia euclidiana es lo suficientemente pequeña.
# Con un umbral de distancia de 0,6, el modelo dlib logró una precisión del 99,38% en el punto de referencia estándar de reconocimiento facial LFW, lo que lo coloca entre los mejores algoritmos para el reconocimiento facial.

# ArcFace
# Este es el modelo más nuevo de la cartera de modelos. Sus diseñadores conjuntos son los investigadores del Imperial College London e InsightFace. 
# El modelo ArcFace alcanza una precisión del 99,40% en el conjunto de datos LFW.

# OpenCV
# Comparado con otros, OpenCV es el detector facial más liviano. Utiliza un algoritmo de haar-cascada que no se basa en técnicas de aprendizaje profundo. Por eso es rápido, pero su rendimiento es 
# relativamente bajo. Para que OpenCV funcione correctamente, se requieren imágenes frontales. Además, su rendimiento de detección ocular es medio. Esto causa problemas de alineación. 
# Tenga en cuenta que el detector predeterminado en DeepFace es OpenCV.

# 2. Algoritimo de Similaridad o Semejanza
# Los modelos de reconocimiento facial son redes neuronales convolucionales regulares y son responsables de representar rostros como vectores. 
# La decisión de verificación se basa en la distancia entre vectores. Podemos clasificar pares si su distancia es menor que un umbral.
# La distancia se puede encontrar mediante diferentes métricas, como la similitud del coseno, la distancia euclidiana y la forma L2. 
# La configuración predeterminada encuentra la similitud del coseno. Alternativamente, puede establecer la métrica de similitud durante la verificación como se muestra a continuación.

# SSD
# SSD son las siglas de Single-Shot Detector; es un detector popular basado en aprendizaje profundo. El rendimiento de SSD es comparable al de OpenCV. Sin embargo, 
# SSD no admite puntos de referencia faciales y depende del módulo de detección de ojos de OpenCV para alinearse. Aunque su rendimiento de detección es alto, 
# la puntuación de alineación es solo media.

# MTCNN
# Este es un detector facial basado en aprendizaje profundo y viene con puntos de referencia faciales. Esa es la razón por la que tanto las puntuaciones de detección como las de alineación 
# son altas para MTCNN. Sin embargo, es más lento que OpenCV, SSD y Dlib.

# RetinaFace
# RetinaFace es reconocido como el modelo de vanguardia basado en el aprendizaje profundo para la detección de rostros. Su desempeño en la naturaleza es desafiante. 
# Sin embargo, requiere una gran potencia de cálculo. Es por eso que RetinaFace es el detector facial más lento en comparación con los demás.

# CONCLUSION: PARAMETROS PARA API VAROPAGO - ONBOARDING - RECONOCIMIENTO FACIAL.
# La plataforma de VaroPago requiere mucha confianza de reconocimiento facial, entonces debemos de usar el detector "RetinaFace" o MTCNN.
# or otro lado, si la alta velocidad fuera es más importante, entonces deberíamos de usar OpenCV o SSD.
# El modelo adoptado aqui es el VGG-Face 97,78% de precisión en el popular conjunto de datos de Caras etiquetadas en la naturaleza (LFW).
# El algoritimo matematico para la similaridad que usaremos es el euclidean_l2 por alta tasa de confiablidad.

# 3. Detectores

# 4. Atributos faciales 
# El Análisis incluyen edad, género, expresión facial (incluyendo enojo, miedo, neutral, triste, disgusto, feliz y sorpresa) 
# y predicciones de raza (incluyendo asiático, blanco, medio oriente, indio, latino y negro). 
# La función de análisis bajo la interfaz de DeepFace se utiliza para encontrar la demografía de una cara.


metrica_similaridad = ["cosine", "euclidean", "euclidean_l2"]
detectores = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]
modelos = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
demografico = ["age", "gender", "race", "emotion"]
database_photos = "photos/" 

#Leer foto via Lib CV2 (OpenCV)
foto = cv2.imread("photos/img2.jpg")
#plt.imshow(foto[:, :, ::-1])

# Deteccion facial y alineamiento del rostro
deteccion_rostro = DeepFace.detectFace(img_path = foto, detector_backend = detectores[4])
plt.imshow(deteccion_rostro)

### 1. Comparando con Pasaport Mexicano ###
reconocimiento_rostro_pasaporte = DeepFace.verify(foto, "id/passporte-mx.png", detector_backend = detectores[4], distance_metric = metrica_similaridad[2], model_name = modelos[0])
# La verificación facial tiene O(1) de complejidad altoritimica
# El reconocimiento facial tiene O(n) de complejidad altoritimica, asi que, el reconocimiento facial se vuelve problemático con las herramientas regulares de verificación facial 
# en datos de nivel de millones / miles de millones y hardware limitado.

#Reconocimiento facial
# El canal de reconocimiento facial moderno consta de 4 etapas comunes: detectar, alinear, representar y verificar. Deepface maneja todas estas etapas comunes en segundo plano. S
# implemente puede llamar a su función de verificación, búsqueda o análisis con una sola línea de código.

#Verificación facial
#Esta función verifica pares de caras como la misma persona o personas diferentes. 
# Podemos pasar un path de imagenes exactas como entradas, asi como, pasamos imágenes codificadas a numpy o based64.

### 2. Comparando con Pasaport INE ###
reconocimiento_rostro_ine = DeepFace.verify(foto, "id/ine.jpg", detector_backend = detectores[4], distance_metric = metrica_similaridad[2], model_name = modelos[0])

### 3. Comparando con VISA USA ###
reconocimiento_rostro_visausa = DeepFace.verify(foto, "id/usa-visa.jpg", detector_backend = detectores[4], distance_metric = metrica_similaridad[2], model_name = modelos[0])


#reconocimiento_rostro_ine = DeepFace.verify(foto, "fm2-plivia.jpg", detector_backend = detectores[4], distance_metric = metrica_similaridad[2], model_name = modelos[0])

### 4. Reconocimiento facial ###
# El reconocimiento facial requiere aplicar la verificación facial muchas veces. Aquí, Deepface tiene una función de búsqueda lista para usar para manejar esta acción. 
# Buscará la identidad de la imagen de entrada en la ruta de la base de datos y devolverá el marco de datos de pandas como salida.
buscar_rostro = DeepFace.find(img_path = foto, db_path = "/Users/gporto-imac/Projects/ai/vision/id", detector_backend = detectores[4], distance_metric = metrica_similaridad[2], model_name = modelos[0])


#for model in modelos:
#    result = DeepFace.verify(foto,"id/ine.jpg", model_name = model)
#    df = DeepFace.find(img_path = foto, db_path = database_photos, model_name = model)

demografia_rostro = DeepFace.analyze(img_path = foto, actions = demografico)


### 5. Streaming y análisis en tiempo real 
# Procesamiento de videos en tiempo real. La función Stream accederá a nuestra cámara web y aplicará tanto el reconocimiento facial como el análisis de atributos faciales. 
# La función comienza a analizar un fotograma si puede enfocar un rostro secuencialmente 5 fotogramas. Luego, muestra los resultados 5 segundos.
# RetinaFace y MTCNN parecen tener un rendimiento superior en las etapas de detección y alineación, pero son más lentos que otros. 
# Si la velocidad de su canalización es más importante, entonces debería usar opencv o ssd. Por otro lado, si considera la precisión, debe usar retinaface o mtcnn.

# Los modelos de reconocimiento facial representan imágenes faciales como incrustaciones vectoriales. La idea detrás del reconocimiento facial es que los vectores deberían ser más similares para 
# la misma persona que para diferentes personas. La pregunta es dónde y cómo almacenar las incrustaciones faciales en un sistema a gran escala. 
# Deepface ofrece una función de representación para encontrar incrustaciones de vectores a partir de imágenes faciales.
embedding = DeepFace.represent(img_path = foto, model_name = modelos[0])


#print("1. Analisis de Reconocimiento facial contra Pasaporte: ", buscar_rostro)
print("1. Analisis de Reconocimiento facial contra Pasaporte: ", reconocimiento_rostro_pasaporte)
print(" Las fotos coinciden con la misma persona del Pasaporte? ",reconocimiento_rostro_pasaporte["verified"])

#print("2. Analisis de Reconocimiento facial contra INE: ", buscar_rostro)
print("2. Analisis de Reconocimiento facial contra INE: ", reconocimiento_rostro_ine)
print(" Las fotos coinciden con misma persona de la credencial INE? ",reconocimiento_rostro_ine["verified"])


#print("3. Analisis de Reconocimiento facial contra INE: ", buscar_rostro)
print("3. Analisis de Reconocimiento facial contra VISA USA: ", reconocimiento_rostro_visausa)
print(" Las fotos coinciden con misma persona de la VISA USA? ",reconocimiento_rostro_visausa["verified"])



print("4. Analisis Demografica del Rostro: ")
print(" Edad: ",demografia_rostro["age"]," años.")
print(" Etnia: ",demografia_rostro["dominant_race"])
print(" Emocion del rostro: ",demografia_rostro["dominant_emotion"])
print(" Sexo: ",demografia_rostro["gender"])

#demografia_rostro_json = json.loads(demografia_rostro)
#print (demografia_rostro_json)