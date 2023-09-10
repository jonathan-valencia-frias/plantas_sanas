import tensorflow as tf



# Verifica las GPUs disponibles
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) == 0:
    print("No se detectaron GPUs.")
else:
    print("GPUs disponibles:")
    for device in physical_devices:
        print(device)