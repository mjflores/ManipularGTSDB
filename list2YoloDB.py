import os
import cv2

'''
train: /path/to/GTSRB/images/train
val: /path/to/GTSRB/images/val

nc: 43
names: [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection',
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
    'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 
    'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
    'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
    'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]
'''

def convert_gtsrb_to_yolo(gtsrb_data, output_dir):
    """
    Convierte los datos de GTSRB al formato YOLO v5.

    :param gtsrb_data: Diccionario con imágenes y etiquetas de GTSRB
    :param output_dir: Directorio de salida para los datos en formato YOLO
    """
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    for i, (image, label) in enumerate(zip(gtsrb_data['images'], gtsrb_data['labels'])):
        # Guardar imagen
        img_path = os.path.join(output_dir, 'images', f'gtsrb_{i:05d}.png')
        cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Crear archivo de etiquetas
        txt_path = os.path.join(output_dir, 'labels', f'gtsrb_{i:05d}.txt')
        with open(txt_path, 'w') as f:
            # GTSRB solo tiene una señal por imagen, centrada
            f.write(f"{label} 0.5 0.5 1.0 1.0\n")


def convert_gtsdb_to_yolo(gtsdb_data, output_dir):
    """
    Convierte los datos de GTSDB al formato YOLO v5.

    :param gtsdb_data: Diccionario con imágenes y anotaciones de GTSDB
    :param output_dir: Directorio de salida para los datos en formato YOLO
    """
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    for i, (image, annotations) in enumerate(zip(gtsdb_data['images'], gtsdb_data['annotations'])):
        # Guardar imagen
        img_path = os.path.join(output_dir, 'images', f'gtsdb_{i:05d}.png')
        cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        #print("annotations",annotations)
        # Crear archivo de etiquetas
        txt_path = os.path.join(output_dir, 'labels', f'gtsdb_{i:05d}.txt')
        with open(txt_path, 'w') as f:
            h, w = image.shape[:2]
            for ann in annotations:
                left, top, right, bottom, class_id = ann
                # Convertir a formato YOLO (normalizado)
                x_center = ((left + right) / 2) / w
                y_center = ((top + bottom) / 2) / h
                width = (right - left) / w
                height = (bottom - top) / h
                f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")

def create_yolo_dataset(gtsrb_data, gtsdb_data, output_dir):
    """
    Crea un conjunto de datos combinado en formato YOLO v5.

    :param gtsrb_data: Datos de GTSRB
    :param gtsdb_data: Datos de GTSDB
    :param output_dir: Directorio de salida
    """
    # Convertir GTSRB
    convert_gtsrb_to_yolo(gtsrb_data, os.path.join(output_dir, 'train'))
    print("Fin 1: convert_gtsrb_to_yolo")
    #print("tamano 1", len(gtsrb_data['images']))
    #print("tamano 2",len(gtsdb_data['images']))
    #print("contenido 2", gtsdb_data)
    # Convertir GTSDB
    convert_gtsdb_to_yolo(gtsdb_data, os.path.join(output_dir, 'val'))
    print("Fin 2: convert_gtsdb_to_yolo")
    # Crear archivo dataset.yaml
    yaml_content = f"""
train: {os.path.join(output_dir, 'train', 'images')}
val: {os.path.join(output_dir, 'val', 'images')}

nc: 43  # número de clases (ajusta según sea necesario)
names: ['speed limit 20', 'speed limit 30', 'speed limit 50', ...]  # añade todos los nombres de clases
"""

    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

