import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os


from config import RUTA_SALIDA_PROCESADA, TAMANO_IMG, BATCH_SIZE, SEED, \
                   NOMBRE_MODELO_GUARDADO 

def load_test_dataset(): 
    print("Cargando el dataset de prueba (o validación si no hay test set dedicado)...")
    test_ds = image_dataset_from_directory(
        RUTA_SALIDA_PROCESADA,
        validation_split=0.2, # Usamos la misma lógica para obtener el subconjunto de validación
        subset="validation",
        seed=SEED,
        image_size=TAMANO_IMG,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    class_names = test_ds.class_names
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    print("Dataset de prueba/validación cargado y optimizado.")
    
    return test_ds, class_names

def load_trained_model(model_path):
    print(f"Cargando modelo desde: {model_path}")
    try:
        model = keras.models.load_model(model_path)
        print("Modelo cargado exitosamente.")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def evaluate_and_report(model, test_ds, class_names):
    print("\nRealizando evaluación final en el conjunto de prueba/validación...")
    loss, accuracy = model.evaluate(test_ds)

    print(f"Pérdida en el conjunto de prueba/validación: {loss:.4f}")
    print(f"Precisión en el conjunto de prueba/validación: {accuracy:.4f}")

    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_true_labels = np.argmax(y_true, axis=1)

    y_pred_probs = model.predict(test_ds)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    print("\nMatriz de Confusión:")
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    print(cm)

    print("\nReporte de Clasificación:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Etiqueta Predicha")
    plt.ylabel("Etiqueta Verdadera")
    plt.title("Matriz de Confusión")
    plt.show()

def plot_training_history(history):
    if history is None:
        print("No hay historial de entrenamiento para graficar.")
        return

    print("\nGenerando gráficas de precisión y pérdida del historial de entrenamiento...")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.title('Precisión del Modelo durante el Entrenamiento')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Pérdida del Modelo durante el Entrenamiento')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(NOMBRE_MODELO_GUARDADO):
        print(f"Error: El modelo no se encontró en '{NOMBRE_MODELO_GUARDADO}'.")
        print("Por favor, entrena el modelo primero ejecutando 'python train.py'.")
    else:
        test_ds, class_names = load_test_dataset()
        model = load_trained_model(NOMBRE_MODELO_GUARDADO)
        if model:
            evaluate_and_report(model, test_ds, class_names)
      