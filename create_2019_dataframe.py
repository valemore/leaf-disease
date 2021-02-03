import pandas as pd
from pathlib import Path
import json
import shutil
from tqdm import tqdm

dir_2019 = Path("2019/cassava-disease/train/")
num2_disease_json = Path("data/label_num_to_disease_map.json")

out_dir = Path("data/2019/")
out_dir.mkdir(exist_ok=True)

with open(num2_disease_json, "r") as f:
    num2disease = json.load(f)

img_ids = []
labels = []

def copy_2019_data(subdir, label):
    subdir.mkdir(exist_ok=True)
    print(f"Copying images in {str(subdir)}...")
    for img_file in tqdm((subdir.glob("*.jpg"))):
        dst = out_dir / img_file.name
        shutil.copy(img_file, dst)
        img_ids.append(img_file.name)
        labels.append(label)

for prefix, label in zip(["cbb", "cbsd", "cgm", "cmd", "healthy"], [0, 1, 2, 3, 4]):
    copy_2019_data(dir_2019 / prefix, label)

labels_df = pd.DataFrame({"image_id": img_ids, "label": labels})
labels_df.to_csv(out_dir / "labels.csv", index=False)