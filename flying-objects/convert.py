from pathlib import Path
from PIL import Image

def convert_images(path):
    path = Path(path)
    output_dir = path / "converted"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for img_path in path.rglob("*.*"):
        if img_path.is_file():
            try:
                img = Image.open(img_path).convert("RGB")
                
                for fmt, ext in [("PNG", "png"), ("BMP", "bmp"), ("JPEG", "jpg")]:
                    relative_path = img_path.relative_to(path).parent
                    target_folder = output_dir / relative_path
                    target_folder.mkdir(parents=True, exist_ok=True)
                    
                    output_file = target_folder / f"{img_path.stem}.{ext}"
                    img.save(output_file, fmt)
                    print(f"Saved: {output_file}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

def organize_by_format(converted_path):
    converted_path = Path(converted_path)
    for img_path in converted_path.rglob("*.*"):  
        if img_path.is_file():
            ext = img_path.suffix.lower().lstrip(".")
            format_folder = converted_path / ext
            format_folder.mkdir(exist_ok=True)  
            
            new_path = format_folder / img_path.name
            img_path.rename(new_path)
            print(f"Moved: {img_path} → {new_path}")

convert_images("/home/alona/універ/3курс/2семестр/digital_image_processing/flying-objects/bird_or_not")
organize_by_format("/home/alona/універ/3курс/2семестр/digital_image_processing/flying-objects/bird_or_not/converted")
