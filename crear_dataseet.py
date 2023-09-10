import os
import pandas as pd

# Rutas de las carpetas
base_folder = "C:\\Users\\jonat\\Desktop\\modular\\Imagenes"
enfermedades_tomates = ["Clavivacter", "Fulvia", "Leveilula", "Pseudomonas", "Xanthomonas"]
enfermedades_pepinos = ["Corynespora", "Mildew"]

# Lista para almacenar los datos
data = []

# Recorre las carpetas de tomates enfermos
for enfermedad in enfermedades_tomates:
    folder_path = os.path.join(base_folder, "tomates enfermos", enfermedad)
    images = os.listdir(folder_path)
    for image in images:
        image_path = os.path.join(folder_path, image)
        data.append((image_path, "tomate_" + enfermedad))

# Recorre la carpeta de pepinos enfermos
for enfermedad in enfermedades_pepinos:
    folder_path = os.path.join(base_folder, "pepinos enfermos", enfermedad)
    images = os.listdir(folder_path)
    for image in images:
        image_path = os.path.join(folder_path, image)
        data.append((image_path, "pepino_" + enfermedad))

# Crear el dataframe
df = pd.DataFrame(data, columns=["ruta_imagen", "etiqueta"])

# Guardar el dataframe en un archivo CSV
df.to_csv("enfermedades_dataset.csv", index=False)
