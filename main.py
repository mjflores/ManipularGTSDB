#import os
#import pandas as pd
#import cv2
#import numpy as np
#from PIL import Image
from gtbd2list import read_gtsrb, read_gtsdb, visualize_images_gtsrb, visualize_images_with_annotations
from list2YoloDB import create_yolo_dataset, convert_gtsrb_to_yolo, convert_gtsdb_to_yolo

# Ejemplo de uso
gtsrb_root = '/home/pc/Descargas/ManipularGTSDB/GTSRB_Final_Training_Images/GTSRB'
#print("Direccion: ",gtsrb_root)
gtsrb_data = read_gtsrb(gtsrb_root)

gtsdb_root = '/home/pc/Descargas/ManipularGTSDB/TrainIJCNN2013'
#print("Direccion: ",gtsdb_root)
gtsdb_data = read_gtsdb(gtsdb_root)


print(f"GTSRB: {len(gtsrb_data['images'])} imágenes cargadas")
print(f"GTSDB: {len(gtsdb_data['images'])} imágenes cargadas")

'''
# Ejemplo de cómo acceder a los datos
print("Ejemplo de GTSRB:")
print(f"Forma de la primera imagen: {gtsrb_data['images'][0].shape}")
print(f"Etiqueta de la primera imagen: {gtsrb_data['labels'][0]}")

print("\nEjemplo de GTSDB:")
print(f"Forma de la primera imagen: {gtsdb_data['images'][0].shape}")
print(f"Anotaciones de la primera imagen: {gtsdb_data['annotations'][0]}")


visualize_images_gtsrb(gtsrb_data['images'], gtsrb_data['labels'], num_images=10)
visualize_images_with_annotations(gtsdb_data, num_images=10)
'''

yolo_output = '/home/pc/Descargas/ManipularGTSDB/yolo_GTSDB/'
create_yolo_dataset(gtsrb_data, gtsdb_data, yolo_output)

print(f"Conjunto de datos YOLO v5 creado en {yolo_output}")