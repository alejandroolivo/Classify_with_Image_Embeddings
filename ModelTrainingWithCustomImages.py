from Util.CustomDataset import CustomImageDataset
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch


# Check if GPU is available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Fine-tune ViT

# Load the dataset
# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225]),  # Normalization values for ImageNet
])

dataset = CustomImageDataset(main_dir='Data/Ropa/Clases', transform=transform)

# Define the hyperparameters
learning_rate = 1e-4
epochs = 50
train_enabled = True

# Fine-tune ViT
if train_enabled:

    # Load the tokenizer for ViT
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    # Load the model for ViT
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=9)
    model = model.to(device)
    
    # Definir tamaño de batch
    batch_size = 16

    # Crear un DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Entrena el modelo
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
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
    model.save_pretrained('./ViT-custom')
    feature_extractor.save_pretrained('./ViT-custom')

    # Print end of training
    print('Training finished')

else:

    # Load the model
    model = ViTForImageClassification.from_pretrained('./ViT-custom')
    model.to(device)

    # Load the tokenizer
    feature_extractor = ViTFeatureExtractor.from_pretrained('./ViT-custom')

    # Test the model
    # Load the image
    image = Image.open('Data/Test/pantalones.png').convert('RGB')

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
    print("Predicted class:", predicted_class_name)

