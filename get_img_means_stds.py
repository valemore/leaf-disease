from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

img_dir = Path("./data/train_images/")
train_imgs = list(img_dir.glob("*"))

n_imgs = len(train_imgs)

channel_means = np.zeros((n_imgs, 3), dtype=float)
channel_vars = np.zeros((n_imgs, 3), dtype=float)

for i, img in tqdm(enumerate(train_imgs)):
    pil_image = Image.open(img)
    img_array = np.array(pil_image) # (600, 800, 3)
    channel_means[i, :] = np.mean(img_array.reshape(-1, 3), axis=0)
    channel_vars[i,:] = np.var(img_array.reshape(-1, 3), axis=0)

channel_means = np.mean(channel_means, axis=0)
channel_vars = np.sqrt(np.mean(channel_vars, axis=0))

with open("img_stats.txt", "w") as f:
    f.write("# Image statistics for train images (without validation images)\n")
    f.write("# Channel means\n")
    f.write(str(channel_means) + "\n")
    f.write("# Channel standard deviations\n")
    f.write(str(channel_vars) + "\n")