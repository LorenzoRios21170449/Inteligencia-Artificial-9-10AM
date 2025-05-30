import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

# Ruta del dataset de entrada y salida
entrada_dir = r"C:\Users\loren\Desktop\Proyecto IA\Dataset sin procesar"
salida_dir = r"C:\Users\loren\Desktop\Proyecto IA\Dataset Procesado"

# Detector Haarcascade frontal
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para aplicar aumentos y normalizar
def preprocesar_imagen(imagen_pil):
    # 1. Brillo aleatorio
    enhancer = ImageEnhance.Brightness(imagen_pil)
    factor_brillo = random.uniform(0.6, 1.4)
    imagen_pil = enhancer.enhance(factor_brillo)

    # 2. Rotación aleatoria
    angulo = random.randint(-20, 20)
    imagen_pil = imagen_pil.rotate(angulo)

    # 3. Escalado aleatorio (zoom)
    scale = random.uniform(0.9, 1.1)
    w, h = imagen_pil.size
    imagen_pil = imagen_pil.resize((int(w * scale), int(h * scale)))

    # 4. Convertir a NumPy y normalizar [0, 1]
    imagen_np = np.array(imagen_pil).astype(np.float32) / 255.0

    return imagen_np

# Recorre carpetas y procesa imágenes
for persona in os.listdir(entrada_dir):
    ruta_persona = os.path.join(entrada_dir, persona)
    if not os.path.isdir(ruta_persona):
        continue

    ruta_salida = os.path.join(salida_dir, persona)
    os.makedirs(ruta_salida, exist_ok=True)

    for img_nombre in os.listdir(ruta_persona):
        ruta_img = os.path.join(ruta_persona, img_nombre)
        imagen_cv = cv2.imread(ruta_img)

        if imagen_cv is None:
            continue

        # Detección de rostro
        gris = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
        rostros = face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5)

        for i, (x, y, w, h) in enumerate(rostros):
            rostro = imagen_cv[y:y+h, x:x+w]
            rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
            imagen_pil = Image.fromarray(rostro_rgb).resize((224, 224))

            imagen_final = preprocesar_imagen(imagen_pil)
            imagen_final = (imagen_final * 255).astype(np.uint8)  # Restaurar para guardar

            img_salida = os.path.join(ruta_salida, f"{os.path.splitext(img_nombre)[0]}_r{i}.jpg")
            Image.fromarray(imagen_final).save(img_salida)

print("✅ Preprocesamiento completado.")
