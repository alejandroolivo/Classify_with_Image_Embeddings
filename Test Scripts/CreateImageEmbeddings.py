from sentence_transformers import SentenceTransformer, util
from PIL import Image

#Load CLIP model
model = SentenceTransformer('clip-ViT-B-32', device='cpu', cache_folder='./cache_models')

#Encode an image:
img_emb = model.encode(Image.open('Data/Test/pantalones.png'))

#Encode text descriptions
text_emb = model.encode(['dress', 'trousers', 't-shirt'])

#Compute cosine similarities 
cos_scores = util.cos_sim(img_emb, text_emb)
print(cos_scores)