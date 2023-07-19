import os
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np
import shutil

#Load CLIP model
model = SentenceTransformer('clip-ViT-B-32', device='cuda:0', cache_folder='./cache_models')

# test
dataset = 'Tornillos'
mode = 'max' # 'max' or 'avg'

#Ruta a la carpeta principal que contiene las carpetas de clases
root_path = './Data/' + dataset + '/Clases'
class_folders = os.listdir(root_path)

#Ruta a la carpeta donde se moverán las imágenes
new_root_path = './Data/' + dataset + '/Dataset'

#Ruta a la carpeta que contiene las imágenes a clasificar
images_folder_path = './Data/' + dataset + '/Images'
image_files = os.listdir(images_folder_path)

for img_file in image_files:
    img_file_path = os.path.join(images_folder_path, img_file)
    
    #Encode an image:
    img_emb = model.encode(Image.open(img_file_path))

    #Crear un diccionario para guardar las similitudes
    similarities = {}

    for folder in class_folders:
        class_path = os.path.join(root_path, folder)
        class_image_files = os.listdir(class_path)
        max_value = 0
        cos_scores_total = 0

        for class_img_file in class_image_files:
            class_img_file_path = os.path.join(class_path, class_img_file)
            class_img_emb = model.encode(Image.open(class_img_file_path))
            
            # Compute cosine similarity and add it to the total
            cos_scores = util.cos_sim(img_emb, class_img_emb)
            cos_scores_total += cos_scores

            if max_value == 0:
                max_value = cos_scores

            #store mas value
            if cos_scores > max_value:
                max_value = cos_scores
        
        if mode == 'max':

            # compute max cosime similarity for this class
            similarities[folder] = max_value

        elif mode == 'avg':            

            # Compute average cosine similarity for this class
            similarities[folder] = cos_scores_total / len(class_image_files)
    
    # Find class with maximum similarity
    max_class = max(similarities, key=similarities.get)
    
    # Create new folder path if it does not exist
    new_folder_path = os.path.join(new_root_path, max_class)
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Move image to new folder
    new_img_file_path = os.path.join(new_folder_path, img_file)
    shutil.move(img_file_path, new_img_file_path)
