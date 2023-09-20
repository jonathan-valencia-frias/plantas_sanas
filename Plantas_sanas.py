import tensorflow as tf
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras import layers, models
from keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

Tamaño_imagen = 250


label_mapping = {
    'tomate_Clavivacter': 0,
    'tomate_Fulvia': 1,
    'tomate_Leveilula': 2,
    'tomate_Pseudomonas': 3,
    'tomate_Xanthomonas': 4,
    'pepino_Corynespora': 5,
    'pepino_Mildew': 6
}

# Cargar el dataframe desde el archivo CSV
df = pd.read_csv("enfermedades_dataset.csv")

# Obtener el número total de muestras en el DataFrame
total_samples = len(df)

# Definir el tamaño del conjunto de prueba
test_size = int(0.1 * total_samples)

# Mezclar aleatoriamente las filas del DataFrame
df = df.sample(frac=1, random_state=42)

# Dividir el DataFrame en conjuntos de entrenamiento y prueba
train_df = df.iloc[:-test_size]
test_df = df.iloc[-test_size:]

# Cargar imágenes y etiquetas
def cargar_imagenes_y_etiquetas(dataframe):
    images = []
    labels = []

    for index, row in dataframe.iterrows():
        image_path = row["ruta_imagen"]
        label = row["etiqueta"]

        # Carga la imagen y realiza cualquier preprocesamiento necesario aquí
        image = cargar_y_preprocesar_imagen(image_path)

        images.append(image)
        labels.append(label)

    return images, labels

# Función para cargar y preprocesar imágenes
def cargar_y_preprocesar_imagen(image_path):
    img = image.load_img(image_path, target_size=(Tamaño_imagen, Tamaño_imagen))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

# Cargar imágenes y etiquetas de entrenamiento y prueba
train_images, train_labels = cargar_imagenes_y_etiquetas(train_df)
test_images, test_labels = cargar_imagenes_y_etiquetas(test_df)

# Función para mapear las etiquetas a valores numéricos
def mapear_etiquetas_a_numeros(etiquetas):
    return [label_mapping[label] for label in etiquetas]

# Convertir las etiquetas de entrenamiento y prueba a valores numéricos
train_labels_numeric = mapear_etiquetas_a_numeros(train_labels)
test_labels_numeric = mapear_etiquetas_a_numeros(test_labels)

# Calcular el número total de clases
num_classes = len(label_mapping)

# Convertir las etiquetas numéricas a one-hot encoding
train_labels = to_categorical(train_labels_numeric, num_classes=num_classes)
test_labels = to_categorical(test_labels_numeric, num_classes=num_classes)

train_images = np.array(train_images)
test_images = np.array(test_images)

# Aplicar preprocess_input a las imágenes
train_images = preprocess_input(train_images)
test_images = preprocess_input(test_images)

# Definir el modelo base (VGG16)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(Tamaño_imagen, Tamaño_imagen, 3))

base_model.trainable = False

## Data augmentation
data_augmentation = tf.keras.Sequential([
#   layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
  layers.RandomContrast(0.2),
])

flatten_layer = layers.Flatten()

# Configuración de las capas densas según 'depths'
dense_units = [8192, 4096]  # Ajusta esta lista según 'dense_units'
depths = [2, 1, 3]  # Ajusta esta lista según 'depths'

dense_layers = []  # Lista para almacenar las capas densas

for depth in depths:
    for units in dense_units:
        dense_layer = layers.Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        dense_layers.append(dense_layer)

prediction_layer = layers.Dense(7, activation='softmax')

model = models.Sequential([
    data_augmentation,
    base_model,
    flatten_layer,
    *dense_layers,  # Agrega todas las capas densas
    prediction_layer
])

# Ejemplo con RMSprop
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=custom_optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

es = EarlyStopping(monitor='loss', mode='min', patience=10, restore_best_weights=True)

model.fit(train_images, train_labels, epochs=10, validation_split=0.3, batch_size=8, callbacks=[es])

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_accuracy)

