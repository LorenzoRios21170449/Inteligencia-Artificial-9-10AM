import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar modelo entrenado
model = load_model('modelo_emociones_transferlearning.h5')

# Clases del dataset
clases = ['Angustiada', 'Enojo', 'Feliz', 'Sorprendida', 'Triste']

# Auxiliar en el clasificador de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# Variables para controlar frecuencia de predicción
contador_frames = 0
intervalo_prediccion = 20  # Aumentar valor = predicciones más espaciadas
ultima_etiqueta = ""
ultima_probabilidad = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gris, 1.1, 4)

    for (x, y, w, h) in rostros:
        margen = int(0.1 * w)
        x1 = max(0, x - margen)
        y1 = max(0, y - margen)
        x2 = min(frame.shape[1], x + w + margen)
        y2 = min(frame.shape[0], y + h + margen)

        if contador_frames % intervalo_prediccion == 0:
            rostro = frame[y1:y2, x1:x2]
            rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
            rostro = cv2.resize(rostro, (224, 224))
            rostro = rostro.astype('float32') / 255.0
            rostro = np.expand_dims(rostro, axis=0)

            preds = model.predict(rostro, verbose=0)
            clase_idx = np.argmax(preds)
            ultima_etiqueta = clases[clase_idx]
            ultima_probabilidad = preds[0][clase_idx]

        contador_frames += 1

        # Muestra el resultado detectado
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        texto = f"{ultima_etiqueta}: {ultima_probabilidad*100:.1f}%"
        cv2.putText(frame, texto, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Detección emociones', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
