import os
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np
import shutil
import json
import time

#Load CLIP model
model = SentenceTransformer('clip-ViT-B-32', device='cuda:0', cache_folder='./cache_models')

# test
dataset = 'Tornillos'
mode = 'avg' # 'max' or 'avg'

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
    
    # start time
    start_time = time.time()    

    #Encode an image:
    img_emb = model.encode(Image.open(img_file_path)).astype(float)

    #Crear un diccionario para guardar las similitudes
    similarities = {}

    for folder in class_folders:
        class_path = os.path.join(root_path, folder)
        class_image_files = os.listdir(class_path)
        max_value = 0
        cos_scores_total = 0

        for class_img_file in class_image_files:
            class_img_file_path = os.path.join(class_path, class_img_file)
            # class_img_emb = model.encode(Image.open(class_img_file_path))

            # Check if the embeddings have been computed before
            embeddings_file_path = class_img_file_path.replace('.png', '.json')
            if os.path.exists(embeddings_file_path):
                with open(embeddings_file_path, 'r') as f:
                    class_img_emb = np.array(json.load(f)).astype(float)
            else:
                class_img_emb = model.encode(Image.open(class_img_file_path)).astype(float)
                with open(embeddings_file_path, 'w') as f:
                    json.dump(class_img_emb.tolist(), f)
            
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
    
    # end time
    end_time = time.time()
    print('Time: ', end_time - start_time)

    # Create new folder path if it does not exist
    new_folder_path = os.path.join(new_root_path, max_class)
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Move image to new folder
    new_img_file_path = os.path.join(new_folder_path, img_file)
    shutil.move(img_file_path, new_img_file_path)
