import os
import pandas as pd
import cv2
import numpy as np
from PIL import Image

'''
Descarga
Recognition
https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
Detection
https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html
'''

def read_gtsrb(root_dir):
    """
        Lee el conjunto de datos GTSRB.

        :param root_dir: Directorio raíz del conjunto de datos GTSRB
        :return: Diccionario con imágenes y etiquetas
    """
    images = []
    labels = []

    # Itera sobre todas las clases
    for c in range(0, 43):
            prefix = os.path.join(root_dir, 'Final_Training', 'Images', format(c, '05d'))
            #print("Salida: ", prefix)

            # Lee el archivo CSV con anotaciones
            gt_path = os.path.join(prefix, 'GT-' + format(c, '05d') + '.csv')
            gt = pd.read_csv(gt_path, sep=';')

            # Itera sobre todas las imágenes en la clase
            for i in range(len(gt)):
                img_path = os.path.join(prefix, gt.iloc[i]['Filename'])
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (32, 32))  # Redimensiona las imágenes a 32x32 píxeles
                images.append(img)
                labels.append(gt.iloc[i]['ClassId'])

    return {'images': np.array(images), 'labels': np.array(labels)}


def read_gtsdb(root_dir):
    """
    Lee el conjunto de datos GTSDB.

    :param root_dir: Directorio raíz del conjunto de datos GTSDB
    :return: Diccionario con imágenes y anotaciones
    """
    images = []
    annotations = []

    # Lee el archivo de anotaciones
    gt = pd.read_csv(os.path.join(root_dir, 'gt.txt'), sep=';',
                     names=['filename', 'left', 'top', 'right', 'bottom', 'class'])

    # Itera sobre todas las imágenes
    for i in range(len(gt)):
        img_path = os.path.join(root_dir, gt.iloc[i, 0])
        img = Image.open(img_path)
        images.append(np.array(img))

        # Extrae las anotaciones
        ann = gt.iloc[i, 1:]#.values
        #print(ann)
        annotations.append([[ann['left'],ann['top'],ann['right'],ann['bottom'],ann['class']]])
        #annotations.append(ann)

    return {'images': np.array(images), 'annotations': np.array(annotations)}

# Función para visualizar imágenes
def visualize_images_gtsrb(images, labels, num_images=10):
    for i in range(num_images):
        img   = images[i]
        label = labels[i]
        cv2.imshow(f'Label: {label}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)  # Espera hasta que se presione una tecla
    cv2.destroyAllWindows()


# Función para visualizar imágenes y anotaciones
def visualize_images_with_annotations(data, num_images=10):
    images = data['images']
    annotations = data['annotations']

    for i in range(num_images):
        img = images[i].copy()
        ann = annotations[i]
        left, top, right, bottom, cls = ann

        # Dibuja la caja delimitadora
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
        cv2.putText(img, str(cls), (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow(f'Class: {cls}', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)  # Espera hasta que se presione una tecla
    cv2.destroyAllWindows()
