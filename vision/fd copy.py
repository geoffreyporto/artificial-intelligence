from deepface import DeepFace
# OpenCV. Comparado con otros, OpenCV es el detector facial más liviano. Utiliza un algoritmo de haar-cascada que no se basa en técnicas de aprendizaje profundo. Por eso es rápido, pero su rendimiento es relativamente bajo. Para que OpenCV funcione correctamente, se requieren imágenes frontales. Además, su rendimiento de detección ocular es medio. Esto causa problemas de alineación. Tenga en cuenta que el detector predeterminado en DeepFace es OpenCV.
# Dlib. Este detector utiliza un algoritmo de cerdo en segundo plano. Por lo tanto, al igual que OpenCV, no se basa en el aprendizaje profundo. Aún así, tiene puntuaciones de detección y alineación relativamente altas.
# SSD. SSD son las siglas de Single-Shot Detector; es un detector popular basado en aprendizaje profundo. El rendimiento de SSD es comparable al de OpenCV. Sin embargo, SSD no admite puntos de referencia faciales y depende del módulo de detección de ojos de OpenCV para alinearse. Aunque su rendimiento de detección es alto, la puntuación de alineación es solo media.
# MTCNN. Este es un detector facial basado en aprendizaje profundo y viene con puntos de referencia faciales. Esa es la razón por la que tanto las puntuaciones de detección como las de alineación son altas para MTCNN. Sin embargo, es más lento que OpenCV, SSD y Dlib.
# RetinaFace. RetinaFace es reconocido como el modelo de vanguardia basado en el aprendizaje profundo para la detección de rostros. Su desempeño en la naturaleza es desafiante. Sin embargo, requiere una gran potencia de cálculo. Es por eso que RetinaFace es el detector facial más lento en comparación con los demás.

detectors = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]

## ssd and opencv = more speed
## retinaface and mtcnn = high confidence

#face verification
verification = DeepFace.verify("img1.jpg", "img2.jpg", detector_backend = detectors[4])
#face recognition
#recognition = DeepFace.find(img_path = "img.jpg", db_path = “C:/facial_db", detector_backend = detectors[0])
#verification = DeepFace.verify(img1_path = "img1.jpg", img2_path = "img2.jpg")

analysis = DeepFace.analyze(img_path = "img.jpg", actions = ["age", "gender", "emotion", "race"])
print(verification)
print(analysis)