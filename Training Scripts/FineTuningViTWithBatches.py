from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from datasets import load_dataset

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Load the tokenizer for ViT
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# Load the model for ViT
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
model = model.to(device)

# Load the dataset
dataset = load_dataset('beans')

# Show one image and its label
ex = dataset['train'][156]

img = ex['image']
# label = ex['label']

# Save the image in current dir
img.save('Data\image.png')

#Show the image
# img.show()

# Preprocess the dataset
def preprocess(examples):
    # Resize the images to 224x224
    images = [np.array(example.resize((224, 224))) for example in examples['image']]
    
    images = np.array(images)
    # Swap the axes to fit the PyTorch format
    images = np.transpose(images, (0, 3, 1, 2))
    labels = np.array(examples['labels'])
    return {'pixel_values': images, 'labels': labels}

dataset = dataset.map(preprocess, batched=True)

# Fine-tune ViT

# Define the hyperparameters
learning_rate = 1e-4
epochs = 3
train_enabled = True

# Fine-tune ViT
if train_enabled:

    # Get the pixel values and the labels
    train_images = torch.tensor(dataset['train']['pixel_values'])
    train_labels = torch.tensor(dataset['train']['labels'])

    train_images = train_images.to(device)
    train_labels = train_labels.to(device)


    # Tokenize the images
    train_encodings = feature_extractor(train_images, return_tensors='pt')
    train_encodings = {key: val.to(device) for key, val in train_encodings.items()}


    # Convertir tus datos en un conjunto de datos de PyTorch
    train_dataset = TensorDataset(train_encodings['pixel_values'], train_labels)

    # Define el tamaño del lote
    batch_size = 16  # Puedes ajustar este valor según tu memoria disponible

    # Usa DataLoader para manejar la división en lotes
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Entrena el modelo
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for batch in train_loader:
            # Desempaqueta el lote
            batch_inputs, batch_labels = batch
            
            # Mueve los datos del lote a la GPU
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            
            # Pasa el lote a través del modelo
            optimizer.zero_grad()
            outputs = model(pixel_values=batch_inputs, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Save the model
    model.save_pretrained('./ViT-beans')

    # Load the model
    model = ViTForImageClassification.from_pretrained('./ViT-beans')

    # Load the tokenizer
    feature_extractor = ViTFeatureExtractor.from_pretrained('./ViT-beans')

# Load the image
image = Image.open('Data/image.png')

# Preprocess the image annd send it to the GPU
inputs = feature_extractor(images=image, return_tensors='pt')
inputs = {key: val.to(device) for key, val in inputs.items()}

# Classify the image
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", dataset['train'].features['labels'].int2str(predicted_class_idx))
