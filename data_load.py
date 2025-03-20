from pathlib import Path
from fastai.vision.all import *
import random
from PIL import Image

def data_load(dataset_path):
    if dataset_path is None:
        return
    
    path = Path(dataset_path)
    
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path, bs=32)

    batch = dls.train.one_batch()
    images = batch[0] 

    return images

def random_image_selection(folder_path):
    path = Path(folder_path)

    if not path.exists() or not path.is_dir():
        print("Шлях не існує або це не папка.")
        return
    
    image_files = list(path.rglob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg")) + list(path.glob("*.bmp"))
    
    if not image_files:
        print("У папці немає зображень.")
        return
    
    random_image_path = random.choice(image_files)
    img = Image.open(random_image_path)
    return img

