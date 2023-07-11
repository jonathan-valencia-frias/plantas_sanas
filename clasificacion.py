import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

categorias=[]
labels=[]
imagenes=[]
categorias=os.listdir('C:\\Users\\hell_\\Documents\\Tareas 2023 A\\Proyecto modular\\Imagenes\\')
print(categorias)
['sana', 'enferma']
x=0
for directorio in categorias:
    for imagen in os.listdir('C:\\Users\\hell_\\Documents\\Tareas 2023 A\\Proyecto modular\\Imagenes\\'+directorio):
        img=Image.open('C:\\Users\\hell_\\Documents\\Tareas 2023 A\\Proyecto modular\\Imagenes\\'+directorio+'\\'+imagen).resize((200,200))
        img=np.asarray(img)
        imagenes.append(img)
        labels.append(x)
    x+=1
print(labels)
[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
imagenes=np.asanyarray(imagenes)
imagenes.shape
(12, 200, 200, 3)
imagenes=imagenes[:,:,:,0]
imagenes.shape
(12, 200, 200)
plt.figure()
plt.imshow(imagenes[9])
plt.colorbar
plt.grid(False)
plt.show()
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(200,200)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
labels=np.asarray(labels)
print(labels)
[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
model.fit(imagenes, labels, epochs=20)
im=0
im=Image.open('C:\\Users\\hell_\\Documents\\Tareas 2023 A\\Proyecto modular\\Imagenes\\Clavibacter\\1.jpg').resize((200,200))
im=np.asarray(im)
im=im[:,:,0]
im=np.array([im])
im.shape
test=im
predicciones=model.predict(test)
print(predicciones)
[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
categorias[np.argmax(predicciones[0])]
