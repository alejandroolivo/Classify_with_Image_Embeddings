import os
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np

#Load CLIP model
model = SentenceTransformer('clip-ViT-B-32', device='cuda:0', cache_folder='./cache_models')

# test
dataset = 'Tornillos'

#Path de la imagen a clasificar
img_path = './Data/' + dataset + '/image.png'

#Encode an image:
img_emb = model.encode(Image.open(img_path))

#Ruta a la carpeta principal que contiene las carpetas de clases
root_path = './Data/' + dataset + '/Clases'

class_folders = os.listdir(root_path)

#Crear un diccionario para guardar las similitudes
similarities = {}

for folder in class_folders:
    class_path = os.path.join(root_path, folder)
    image_files = os.listdir(class_path)
    
    cos_scores_total = 0
    for img_file in image_files:
        img_file_path = os.path.join(class_path, img_file)
        class_img_emb = model.encode(Image.open(img_file_path))
        
        # Compute cosine similarity and add it to the total
        cos_scores = util.cos_sim(img_emb, class_img_emb)
        cos_scores_total += cos_scores
    
    # Compute average cosine similarity for this class
    similarities[folder] = cos_scores_total / len(image_files)

# Calculate total sum of similarities
total_sum = sum(similarities.values())

# Convert raw similarity scores to percentages
for key in similarities.keys():
    similarities[key] = (similarities[key] / total_sum) * 100

# Print the results sorted by score
similarities = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)}
print(similarities)
