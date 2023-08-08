from Util.CustomDataset import CustomImageDataset
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from torch import nn

# Check if GPU is available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

proyecto = 'Ropa'

# Fine-tune ViT

# Load the dataset
# Define the transformation
transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
])

dataset = CustomImageDataset(main_dir='Data/' + proyecto + '/Clases', transform=transform)

# Define the hyperparameters
learning_rate = 1e-5
epochs = 10
train_enabled = True
retrain_enabled = True

# Fine-tune ViT
if train_enabled:
    
    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    if(retrain_enabled):
        # Load the model
        feature_extractor = ViTFeatureExtractor.from_pretrained('./ViT-custom')

        model = ViTForImageClassification.from_pretrained('./ViT-custom')
        model.to(device)

    else:

        # Load the tokenizer for ViT
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

        # Load the model for ViT
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=9)
        model = model.to(device)
    
    # Definir tamaño de batch
    batch_size = 32

    # Crear un DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Entrena el modelo
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        for batch in train_loader:
            batch_images, batch_labels = batch
            batch_inputs = feature_extractor(batch_images, return_tensors="pt")
            pixel_values = batch_inputs["pixel_values"].to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=batch_labels)
            logits = outputs.logits

            loss = criterion(logits, batch_labels)

            print(f'Loss: {loss.item()} for epoch {epoch + 1}')
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
    image = Image.open('Data/Test/camisa.png').convert('RGB')

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

