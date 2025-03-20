from fastai.vision.all import *
from visualization import show_images, show_info, show_hist, show_filters, show_neighbors
from data_load import data_load, random_image_selection
from editing import quantize_image

def main():
    while True:
        print("\nChoose option:")
        print("1 - Set dataset")
        print("2 - Get photo from camera")
        print("0 - Exit")

        choice = input("Your choice: ")

        if choice == "1":
            dataset_path = set_dataset()
            images = data_load(dataset_path)
            show_images(images, N=32, title="Batch of Images")
            img = random_image_selection(dataset_path)
            image_operation(img)
        elif choice == "0":
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 0.")

def set_dataset():
    while True:
        print("\nChoose file format for dataset:")
        print("1 - JPG")
        print("2 - PNG")
        print("3 - BMP")
        print("0 - Exit")

        choice = input("Your choice: ")
        dataset_path = None

        if choice == "1":
            dataset_path = "/home/alona/універ/3курс/2семестр/digital_image_processing/flying-objects/bird_or_not/jpg"
        elif choice == "2":
            dataset_path = "/home/alona/універ/3курс/2семестр/digital_image_processing/flying-objects/bird_or_not/png"
        elif choice == "3":
            dataset_path = "/home/alona/універ/3курс/2семестр/digital_image_processing/flying-objects/bird_or_not/bmp"
        elif choice == "0":
            print("Exiting program...")
            return None
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 0.")
            continue

        print(f"Selected dataset: {dataset_path}")
        return dataset_path
    
def image_operation(img):
    while True:
        print("\nChoose img operation:")
        print("1 - Information")
        print("2 - Histogram")
        print("3 - Filters")
        print("4 - Neighbors pixels")
        print("5 - Quantize")
        print("0 - Exit")

        choice = input("Your choice: ")
        dataset_path = None

        if choice == "1":
            show_info(img)
            continue
        elif choice == "2":
            show_hist(img)
            continue
        elif choice == "3":
            show_filters(img)
            continue
        elif choice == "4":
            show_neighbors(img, 100, 100, connectivity=4)
            show_neighbors(img, 100, 100, connectivity=8)
            continue
        elif choice == "5":
            quantized_image = quantize_image(img, color_depth=8)
            show_info(quantized_image, title="Quantized")
            continue
        elif choice == "0":
            print("Exiting program...")
            return None
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 0.")
            continue

if __name__ == "__main__":
    main()

