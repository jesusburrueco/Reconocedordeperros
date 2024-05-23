import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle
import os
from google.colab import drive
import shutil

print("TensorFlow version:", tf.__version__)

# Montar Google Drive
drive.mount('/content/drive')

# Definir funciones para guardar y cargar el modelo y el historial
def save_model_and_history(model, history, model_path, history_path):
    model.save(model_path)
    with open(history_path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def load_model_and_history(model_path, history_path):
    model = tf.keras.models.load_model(model_path)
    with open(history_path, 'rb') as file_pi:
        history = pickle.load(file_pi)
    return model, history

# Verificar si el modelo y el historial existen
model_path = '/content/drive/My Drive/my_model.h5'
history_path = '/content/drive/My Drive/trainHistoryDict'

def analyze_results(history):
    # Resumen del Modelo
    print("Resumen del Modelo:")
    model.summary()

    # Resultados del Entrenamiento
    history_dict = history.history
    epochs = range(len(history_dict['loss']))

    print("\nResultados del Entrenamiento:")
    for i, epoch in enumerate(epochs):
        print(f"Epoch {i+1}")
        print(f"  - Loss: {history_dict['loss'][i]}")
        print(f"  - Accuracy: {history_dict['accuracy'][i]}")
        print(f"  - Val Loss: {history_dict['val_loss'][i]}")
        print(f"  - Val Accuracy: {history_dict['val_accuracy'][i]}")

    # Gráficos de Pérdida y Precisión
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict['loss'], 'r', label='Training loss')
    plt.plot(epochs, history_dict['val_loss'], 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict['accuracy'], 'r', label='Training accuracy')
    plt.plot(epochs, history_dict['val_accuracy'], 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def train_model():
    # Crear el directorio donde almacenaremos las imágenes
    data_dir = 'manual_upload'
    os.makedirs(data_dir, exist_ok=True)

    # Crear subdirectorios para las razas (asegúrate de que coinciden con los que se usaron durante el entrenamiento)
    breeds = ['beagle', 'labrador', 'bodeguero', 'pitbull']
    for breed in breeds:
        os.makedirs(os.path.join(data_dir, breed), exist_ok=True)

    # Subir imágenes manualmente
    from google.colab import files
    uploaded = files.upload()

    # Guardar las imágenes subidas en el directorio correspondiente
    for filename in uploaded.keys():
        print(f"Nombre del archivo: {filename}")
        for i, breed in enumerate(breeds):
            print(f"{i}: {breed}")
        label = int(input("Ingrese el número correspondiente a la raza: "))
        shutil.move(filename, os.path.join(data_dir, breeds[label], filename))

    # Generador de datos que usa el archivo CSV
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        seed=42
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        seed=42
    )

    # Verificar que hay suficientes imágenes en los generadores
    if train_generator.samples == 0 or validation_generator.samples == 0:
        raise ValueError("No hay suficientes imágenes en las divisiones de entrenamiento o validación. Asegúrate de subir suficientes imágenes y que estén etiquetadas correctamente.")

    # Determinar el número de clases a partir del generador de entrenamiento
    num_classes = len(train_generator.class_indices)

    # Crear el modelo
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
        epochs=10,
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // validation_generator.batch_size)
    )

    save_model_and_history(model, history, model_path, history_path)
    analyze_results(history)

# Función para predecir la clase de una imagen y etiquetarla manualmente
def predict_and_label_image(img_path, data_dir):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_indices_inv = {v: k for k, v in class_indices.items()}
    predicted_class = class_indices_inv[np.argmax(prediction)]
    print(f"Predicción: La imagen es de un {predicted_class}")

    print("\nIngrese la raza correcta de la imagen:")
    for i, breed in enumerate(class_indices.keys()):
        print(f"{i}: {breed}")
    correct_label = int(input("Ingrese el número correspondiente a la raza: "))
    correct_breed = list(class_indices.keys())[correct_label]

    # Mover la imagen a la carpeta correspondiente
    shutil.move(img_path, os.path.join(data_dir, correct_breed, os.path.basename(img_path)))
    print(f"La imagen ha sido movida a la carpeta '{correct_breed}'.")

# Verificar si el modelo y el historial existen
data_dir = 'manual_upload'
if os.path.exists(model_path) and os.path.exists(history_path):
    model, history = load_model_and_history(model_path, history_path)
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        seed=42
    )
    class_indices = train_generator.class_indices
else:
    model = None
    history = None
    class_indices = None

# Menú principal
def main():
    while True:
        print("Seleccione una opción:")
        print("1. Entrenar modelo")
        print("2. Predecir y etiquetar imagen")
        print("3. Salir")
        choice = input("Ingrese el número de su elección: ")

        if choice == '1':
            train_model()
        elif choice == '2':
            if model is None or class_indices is None:
                print("El modelo no está entrenado. Por favor, entrene el modelo primero.")
            else:
                img_path = input("Ingrese la ruta de la imagen a predecir: ")
                predict_and_label_image(img_path, data_dir)
        elif choice == '3':
            break
        else:
            print("Opción inválida. Por favor, intente de nuevo.")

main()
