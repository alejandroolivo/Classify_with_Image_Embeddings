import os
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np
import shutil
import time
import torch

av_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Carga del modelo CLIP
model = SentenceTransformer('clip-ViT-B-32', device=av_device, cache_folder='./cache_models')

# Parámetros del dataset
dataset = 'Ropa'

# Ruta a la carpeta principal que contiene las carpetas de clases
root_path = './Data/' + dataset + '/Clases'
class_folders = os.listdir(root_path)

# Generamos embeddings de texto a partir de los nombres de las carpetas de clases
text_emb = model.encode(class_folders).astype(float)

# Ruta a la carpeta donde se moverán las imágenes
new_root_path = './Data/' + dataset + '/Dataset'

# Ruta a la carpeta que contiene las imágenes a clasificar
images_folder_path = './Data/' + dataset + '/Images'
image_files = os.listdir(images_folder_path)

for img_file in image_files:
    img_file_path = os.path.join(images_folder_path, img_file)
    
    # start time
    start_time = time.time()    

    # Encode an image
    img_emb = model.encode(Image.open(img_file_path)).astype(float)
    
    # Compute cosine similarities between image embedding and text embeddings
    cos_scores = util.cos_sim(img_emb, text_emb)

    # Find the index with the maximum similarity score
    max_index = np.argmax(cos_scores)
    
    # Get the corresponding class name
    max_class = class_folders[max_index]
    
    # end time
    end_time = time.time()
    print('Time: ', end_time - start_time)

    # Create new folder path if it does not exist
    new_folder_path = os.path.join(new_root_path, max_class)
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Move image to the new folder
    new_img_file_path = os.path.join(new_folder_path, img_file)
    shutil.move(img_file_path, new_img_file_path)
