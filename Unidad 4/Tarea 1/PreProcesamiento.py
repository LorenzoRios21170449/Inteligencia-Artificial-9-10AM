import os
import cv2
import numpy as np

from config import RUTA_ORIGINAL, RUTA_SALIDA_PROCESADA, TAMANO_IMG, \
                   GRADOS_ROTACION_MAX, FACTOR_ZOOM, \
                   FACTOR_ILUMINACION_MIN, FACTOR_ILUMINACION_MAX

def ajustar_iluminacion(imagen):
    # Los factores de iluminación se obtienen de config.py
    factor = np.random.uniform(FACTOR_ILUMINACION_MIN, FACTOR_ILUMINACION_MAX)
    return np.clip(imagen * factor, 0, 255).astype(np.uint8)

def rotar_imagen(imagen): 
    h, w = imagen.shape[:2]
    # Los grados máximos de rotación se obtienen de config.py
    angulo = np.random.uniform(-GRADOS_ROTACION_MAX, GRADOS_ROTACION_MAX)
    M = cv2.getRotationMatrix2D((w//2, h//2), angulo, 1)
    return cv2.warpAffine(imagen, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def escalar_imagen(imagen): 
    h, w = imagen.shape[:2]
    # El factor de zoom se obtiene de config.py
    factor = np.random.uniform(1 - FACTOR_ZOOM, 1 + FACTOR_ZOOM)
    imagen_zoom = cv2.resize(imagen, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    
    # Recortar o rellenar imagen para que mantenga el TAMANO_IMG
    if factor > 1: 
        startx = (imagen_zoom.shape[1] - w) // 2
        starty = (imagen_zoom.shape[0] - h) // 2
        return imagen_zoom[starty:starty + h, startx:startx + w]
    else: # Si la imagen se encogió (zoom out), rellenar con ceros
        nueva = np.zeros((h, w, 3), dtype=np.uint8)
        startx = (w - imagen_zoom.shape[1]) // 2
        starty = (h - imagen_zoom.shape[0]) // 2
        nueva[starty:starty + imagen_zoom.shape[0], startx:startx + imagen_zoom.shape[1]] = imagen_zoom
        return nueva

def normalizar_pixeles(imagen): 
    return imagen.astype(np.float32) / 255.0

def preprocesar_imagen_completa(imagen): 

    imagen_aumentada = ajustar_iluminacion(imagen)
    imagen_aumentada = rotar_imagen(imagen_aumentada)
    imagen_aumentada = escalar_imagen(imagen_aumentada)
    
    # para asegurar que todas las imágenes al fina tengan el mismo tamaño.
    imagen_final = cv2.resize(imagen_aumentada, TAMANO_IMG, interpolation=cv2.INTER_AREA)
    
    # Normalizar los píxeles
    imagen_normalizada = normalizar_pixeles(imagen_final)
    
    return imagen_normalizada

def procesar_dataset():

    print(f"Iniciando preprocesamiento de imágenes desde: {RUTA_ORIGINAL}")
    print(f"Las imágenes procesadas se guardarán en: {RUTA_SALIDA_PROCESADA}")

    total_imagenes_procesadas = 0
    total_imagenes_saltadas = 0 

    # Crear el directorio de salida si no existe
    os.makedirs(RUTA_SALIDA_PROCESADA, exist_ok=True)

    for clase in os.listdir(RUTA_ORIGINAL):
        ruta_clase = os.path.join(RUTA_ORIGINAL, clase)
        ruta_salida_clase = os.path.join(RUTA_SALIDA_PROCESADA, clase)
        os.makedirs(ruta_salida_clase, exist_ok=True)

        if not os.path.isdir(ruta_clase):
            continue # Saltar si no es un directorio (ej. .DS_Store)

        for archivo in os.listdir(ruta_clase):
            ruta_img = os.path.join(ruta_clase, archivo)
            
            # Solo procesar archivos de imagen comunes
            if not archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                continue

            imagen = cv2.imread(ruta_img)
            if imagen is None:
                print(f"Error: No se pudo cargar la imagen {ruta_img}")
                total_imagenes_saltadas += 1
                continue

            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            
            # Llamar a la nueva función de preprocesamiento
            imagen_proc = preprocesar_imagen_completa(imagen_rgb)
            
            # Guardar imagen preprocesada (de 0-1 a 0-255 y de RGB a BGR para guardar)
            imagen_guardar = (imagen_proc * 255).astype(np.uint8)
            salida_path = os.path.join(ruta_salida_clase, archivo)
            cv2.imwrite(salida_path, cv2.cvtColor(imagen_guardar, cv2.COLOR_RGB2BGR))
            total_imagenes_procesadas += 1

    print(f"\nPreprocesamiento terminado. {total_imagenes_procesadas} imágenes procesadas, {total_imagenes_saltadas} imágenes saltadas.")

if __name__ == "__main__":

    procesar_dataset()