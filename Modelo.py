import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import TAMANO_IMG # Importa el tamaño de imagen de config

def create_emotion_model(num_classes):

    print(f"Creando modelo ResNet50 para {num_classes} clases...")

    # Cargar el modelo base pre-entrenado (ResNet50)
    base_model = keras.applications.ResNet50(
        input_shape=TAMANO_IMG + (3,), # (224, 224, 3) para imágenes RGB
        include_top=False, # No incluimos las capas de clasificación finales de ImageNet
        weights='imagenet' # Usamos los pesos pre-entrenados en ImageNet
    )

    # Congelar las capas del modelo base
    # Esto evita que los pesos de ResNet50 se actualicen durante el entrenamiento inicial
    base_model.trainable = False

    # Construir el nuevo modelo 
    inputs = keras.Input(shape=TAMANO_IMG + (3,))
    
    # Pasar las entradas a través del modelo base.
    # 'training=False' es importante para que las capas de BatchNormalization del modelo base
    x = base_model(inputs, training=False)

    # Añadir una capa GlobalAveragePooling2D para aplanar las características
    x = layers.GlobalAveragePooling2D()(x)
    
    # Añadir una capa Dropout para regularización
    x = layers.Dropout(0.2)(x)
    
    # Añadir la capa de salida con 'softmax' para clasificación multiclase
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Crear el modelo final
    model = keras.Model(inputs, outputs)

    print("Modelo ResNet50 creado:")
    model.summary()
    
    return model

if __name__ == "__main__":
    dummy_model = create_emotion_model(num_classes=5)
    # No se entrenará aquí, solo se mostrará el resumen.