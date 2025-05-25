import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.model_selection import train_test_split

from config import RUTA_SALIDA_PROCESADA, TAMANO_IMG, BATCH_SIZE, \
                   EPOCHS, LEARNING_RATE, SEED, NOMBRE_MODELO_GUARDADO
from Modelo import create_emotion_model

def load_full_dataset():
    print("Cargando dataset completo sin división...")
    full_ds = keras.preprocessing.image_dataset_from_directory(
        RUTA_SALIDA_PROCESADA,
        image_size=TAMANO_IMG,
        batch_size=1,  # batch=1 para cargar todas las imágenes individualmente
        shuffle=True,
        seed=SEED,
        label_mode='categorical'
    )

    images = []
    labels = []

    for img, label in full_ds:
        images.append(img[0].numpy())
        labels.append(label[0].numpy())

    images = np.array(images)
    labels = np.array(labels)
    print(f"Dataset cargado con {len(images)} imágenes.")
    return images, labels, full_ds.class_names

def split_dataset(images, labels):
    # Dividir en train (70%) y temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    # Dividir temp en val (15%) y test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    print(f"Split de datos: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def prepare_tf_dataset(images, labels, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    if augment:
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.2)
        ])
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds

def compile_and_train_model(model, train_ds, val_ds):
    print("\nCompilando modelo...")
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Entrenando modelo...")
    history = model.fit(train_ds,
                        epochs=EPOCHS,
                        validation_data=val_ds)
    print("Entrenamiento finalizado.")
    return history

def save_model(model, path):
    model.save(path)
    print(f"Modelo guardado en: {path}")

if __name__ == "__main__":
    images, labels, class_names = load_full_dataset()
    num_classes = len(class_names)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(images, labels)

    train_ds = prepare_tf_dataset(X_train, y_train, augment=True)
    val_ds = prepare_tf_dataset(X_val, y_val, augment=False)
    test_ds = prepare_tf_dataset(X_test, y_test, augment=False)

    model = create_emotion_model(num_classes)
    history = compile_and_train_model(model, train_ds, val_ds)
    save_model(model, NOMBRE_MODELO_GUARDADO)

    # Opcional: evaluar en test set
    print("\nEvaluando modelo en test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
