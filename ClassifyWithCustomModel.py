import os
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import shutil
import time
from Util.CustomDataset import CustomImageDataset
import torch

# test
proyecto = 'Ropa'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
])

dataset = CustomImageDataset(main_dir='Data/' + proyecto + '/Clases', transform=transform)

#Ruta a la carpeta principal que contiene las carpetas de clases
root_path = './Data/' + proyecto + '/Clases'
class_folders = os.listdir(root_path)

#Ruta a la carpeta donde se moverán las imágenes
new_root_path = './Data/' + proyecto + '/Dataset'

#Ruta a la carpeta que contiene las imágenes a clasificar
images_folder_path = './Data/' + proyecto + '/Images'
image_files = os.listdir(images_folder_path)

# Load the model
model = ViTForImageClassification.from_pretrained('./ViT-custom')
model.to(device)

# Load the tokenizer
feature_extractor = ViTFeatureExtractor.from_pretrained('./ViT-custom')

# Test the model
for img_file in image_files:
    img_file_path = os.path.join(images_folder_path, img_file)
    
    # start time
    start_time = time.time()    

    # Load the image
    image = Image.open(img_file_path).convert('RGB')

    # Preprocess the image
    image = transform(image)

    # Preprocess the image annd send it to the GPU
    inputs = feature_extractor(images=image, return_tensors='pt')
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Classify the image
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_name = dataset.get_class_name(predicted_class_idx)
    print("Predicted class for " + img_file + ":", predicted_class_name)
    
    # end time
    end_time = time.time()
    print('Time: ', end_time - start_time)

    # Create new folder path if it does not exist
    new_folder_path = os.path.join(new_root_path, predicted_class_name)
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Move image to new folder
    new_img_file_path = os.path.join(new_folder_path, img_file)
    shutil.move(img_file_path, new_img_file_path)
