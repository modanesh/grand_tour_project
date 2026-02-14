import torch
from transformers import pipeline
from dotenv import load_dotenv
import numpy as np

load_dotenv()


from transformers.image_utils import load_image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

feature_extractor = pipeline(
    model="facebook/dinov2-small",
    task="image-feature-extraction", 
)
features = feature_extractor(image)
print(np.asarray(features).shape)
