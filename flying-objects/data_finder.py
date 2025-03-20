from duckduckgo_search import DDGS
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *
import time

def search_images(keywords, max_images=500):
    with DDGS() as ddgs:
        results = ddgs.images(keywords, max_results=max_images)
        return L(results).itemgot('image')

searches = ['bird', 'plane', 'helicopter', 'rocket', 'drones', 'airships']
path = Path('bird_or_not')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    print(f'Завантаження для категорії: {o}')
    urls = search_images(f'{o} photo', max_images=400)
    download_images(dest, urls=urls)
    time.sleep(5)
    resize_images(dest, max_size=400, dest=dest)

failed = verify_images(get_image_files(path))
print(f"Кількість пошкоджених: {len(failed)}")
failed.map(Path.unlink)

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)
