from sentence_transformers import SentenceTransformer, util
from PIL import Image

#Load CLIP model
model = SentenceTransformer('clip-ViT-B-32', device='cuda:0', cache_folder='./cache_models')

#Encode an image:
img_emb = model.encode(Image.open('Data/Pollos (10).png'))

#Encode text descriptions
text_emb = model.encode(['six plucked chickens', 'two plucked chickens', 'eight plucked chickens'])

#Compute cosine similarities 
cos_scores = util.cos_sim(img_emb, text_emb)
print(cos_scores)