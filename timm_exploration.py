import timm
from pprint import pprint
model_names = timm.list_models(pretrained=True)
pprint(model_names)

m = timm.create_model('efficientnet_b0', pretrained=True)
m.eval()

# input resolution 224x224
from torchvision import transforms

import os
from pathlib import Path

img_dir = Path("./data/train_images/")
n_images = 21397

import random
img_dir.glob("*")

