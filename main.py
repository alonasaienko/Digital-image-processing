from tkinter import Image

import numpy as np
from fastai.vision.all import *
from visualization import show_filter_results, show_images, show_info, show_hist, show_filters, show_neighbors, show_contrast
from data_load import data_load, random_image_selection
from editing import add_gaussian_noise, back_filter, back_filter_rgb, box_filter, constrained_least_squares, constrained_least_squares_rgb, gaussian_filter, imp_noise, median_filter, quantize_image, wiener_filter

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
            quantized_4 = quantize_image(img, 4)
            quantized_8 = quantize_image(img, 8)
            show_filter_results(img, [quantized_4, quantized_8],
                            ['4 colors', '8 colors'])
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
            continue
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
        print("2 - Show all filters: Median, Box, Wiener")
        print("3 - Back filter")
        print("4 - Constrained least squares")
        print("0 - Exit")

        choice = input("Your choice: ")

        if choice == "1":
            gaussian_noise = add_gaussian_noise(img, sigma=25)
            impulse_noise = imp_noise(img)
            show_filter_results(img, [gaussian_noise, impulse_noise], 
                              ['Gaussian Noise', 'Impulse Noise'])
            continue
        elif choice == "2":
            noisy_img = add_gaussian_noise(img, sigma=25)

            wiener = wiener_filter(noisy_img)
            median = median_filter(noisy_img)
            mean = box_filter(noisy_img)

            show_filter_results(noisy_img, [wiener, median, mean], 
                              ['Wiener Filter', 'Median Filter', 'Mean(Box) Filter'])
            continue
        elif choice == "3":
            blurred_img = gaussian_filter(img, radius=3)
            restored_img = back_filter_rgb(blurred_img)
            show_filter_results(img, [blurred_img, restored_img],
                              ['Blurred', 'Restored'])
            continue
        elif choice == "4":
            blurred_img = gaussian_filter(img, radius=3)
            cls_restored = constrained_least_squares_rgb(blurred_img)
            show_filter_results(img, [blurred_img, cls_restored],
                            ['Blurred', 'Restored (Constrained LSQ)'])
            continue
        elif choice == "0":
            print("Exiting program...")
            return None
        else:
            print("Invalid choice. Please enter a number between 0 and 10.")

#Реалізувати конвертація між кольоровими моделями rgb hsv cmy ycbcr
#Виконати згладжування кольорового зображення
#Реалізувати підвищення різкості кольорового зображення
#Реалізувати сегментацію на основі кольору (кластеризація, порогова обробка)
#Виконати стиснення зображення за допомогою кольорових можелей jpeg
def lab_4(img):
    while True:
        print("\nChoose img operation:")
        print("1 - Converting to different color model")
        print("2 - Show all filters: Median, Box, Wiener")
        print("3 - Back filter")
        print("4 - Constrained least squares")
        print("0 - Exit")

        choice = input("Your choice: ")

        if choice == "1":
            gaussian_noise = add_gaussian_noise(img, sigma=25)
            impulse_noise = imp_noise(img)
            show_filter_results(img, [gaussian_noise, impulse_noise], 
                              ['Gaussian Noise', 'Impulse Noise'])
            continue
        elif choice == "2":
            noisy_img = add_gaussian_noise(img, sigma=25)

            wiener = wiener_filter(noisy_img)
            median = median_filter(noisy_img)
            mean = box_filter(noisy_img)

            show_filter_results(noisy_img, [wiener, median, mean], 
                              ['Wiener Filter', 'Median Filter', 'Mean(Box) Filter'])
            continue
        elif choice == "3":
            blurred_img = gaussian_filter(img, radius=3)
            restored_img = back_filter_rgb(blurred_img)
            show_filter_results(img, [blurred_img, restored_img],
                              ['Blurred', 'Restored'])
            continue
        elif choice == "4":
            blurred_img = gaussian_filter(img, radius=3)
            cls_restored = constrained_least_squares_rgb(blurred_img)
            show_filter_results(img, [blurred_img, cls_restored],
                            ['Blurred', 'Restored (Constrained LSQ)'])
            continue
        elif choice == "0":
            print("Exiting program...")
            return None
        else:
            print("Invalid choice. Please enter a number between 0 and 10.")
    
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

