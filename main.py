from fastai.vision.all import *
from visualization import show_images, show_info, show_hist, show_filters, show_neighbors, show_contrast, show_noise, show_back_blur
from data_load import data_load, random_image_selection
from editing import quantize_image

def main():
    while True:
        print("\nChoose option:")
        print("1 - Set dataset")
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
    
def lab_1(img):
    while True:
        print("\nChoose img operation:")
        print("1 - Neighbors pixels")
        print("2 - Quantize")
        print("0 - Exit")

        choice = input("Your choice: ")

        if choice == "1":
            show_neighbors(img, 100, 100, connectivity=4)
            show_neighbors(img, 100, 100, connectivity=8)
            continue
        elif choice == "2":
            quantized_image = quantize_image(img, color_depth=8)
            show_info(quantized_image, title="Quantized")
            continue
        elif choice == "0":
            print("Exiting program...")
            return None
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 0.")
            continue

def lab_2(img):
    while True:
        print("\nChoose img operation:")
        print("1 - Histogram")
        print("2 - Filters")
        print("3 - Contrast correction")
        print("0 - Exit")

        choice = input("Your choice: ")

        if choice == "1":
            show_hist(img)
            continue
        elif choice == "2":
            show_filters(img)
            continue
        elif choice == "3":
            show_contrast(img)
            show_contrast(img.convert('L'))
        elif choice == "0":
            print("Exiting program...")
            return None
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 0.")
            continue

def lab_3(img):
    while True:
        print("\nChoose img operation:")
        print("1 - Adding noise")
        print("2 - Quantize")
        print("3 - Back blur")
        print("0 - Exit")

        choice = input("Your choice: ")

        if choice == "1":
            show_noise(img)
            continue
        elif choice == "2":
            quantized_image = quantize_image(img, color_depth=8)
            show_info(quantized_image, title="Quantized")
            continue
        elif choice == "3":
            show_back_blur(img)
        elif choice == "0":
            print("Exiting program...")
            return None
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 0.")
            continue
    
def image_operation(img):
    while True:
        print("\nChoose img operation:")
        print("0 - Information")
        print("1 - Lab1")
        print("2 - Lab2")
        print("3 - Lab3")
        print("10 - Exit")

        choice = input("Your choice: ")

        if choice == "0":
            show_info(img)
            continue
        elif choice == "1":
            lab_1(img)
            continue
        elif choice == "2":
            lab_2(img)
            continue
        elif choice == "3":
            lab_3(img)
            continue
        elif choice == "10":
            print("Exiting program...")
            return None
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 0.")
            continue

if __name__ == "__main__":
    main()

