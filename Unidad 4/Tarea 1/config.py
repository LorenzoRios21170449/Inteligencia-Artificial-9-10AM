import cv2
import os

#Rutas de los dataset usados
RUTA_ORIGINAL = r"C:\Users\cesar\OneDrive\Escritorio\IA\vison artificial\dataset"
RUTA_SALIDA_PROCESADA = r"C:\Users\cesar\OneDrive\Escritorio\IA\vison artificial\dataset_Procesadas"
RUTA_HAAR_CASCADE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

#Parámetros de Imagen
TAMANO_IMG = (224, 224)  # Tamaño al que se redimensionarán las imágenes (usado en  elmodelo  y preprocesamiento)

#Parametros de aumento de datos
GRADOS_ROTACION_MAX = 15
FACTOR_ZOOM = 0.1
FACTOR_ILUMINACION_MIN = 0.6
FACTOR_ILUMINACION_MAX = 1.4

#Valores para entrenamiento
BATCH_SIZE = 32
EPOCHS = 20  # Aumentado para mejor aprendizaje
LEARNING_RATE = 0.0001
SEED = 123  # Para reproducibilidad

# División de dataset en porcentajes (porcentajes en decimales)
FRAC_TRAIN = 0.70
FRAC_VAL = 0.15
FRAC_TEST = 0.15

#Ruta donde se guardara el dataset procesado
NOMBRE_MODELO_GUARDADO = "modelo_reconocimiento_emociones_ResNet50.keras"
