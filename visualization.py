import matplotlib.pyplot as plt
from fastai.vision.all import *
import torchvision
from PIL import Image
import numpy as np

from editing import gaussian_filter, get_histogram, equalize_histogram, median_filter, box_filter, sharpen_image, linear_contrast, get_neighbors

def show_images(images, N=32, title=None):
    plt.figure(figsize=(8, 8))
    
    if images.is_cuda:
        images = images.cpu()

    grid = torchvision.utils.make_grid(images, nrow=8)
    
    grid = grid.numpy().transpose((1, 2, 0))
    
    plt.imshow(grid)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def show_info(images, title="Image Analysis"):

    if not isinstance(images, list):
        images = [images]

    n = len(images)
    plt.figure(figsize=(6 * n, 6 * 2 ))

    for i, img in enumerate(images):
        size = f"Size: {img.size[0]} x {img.size[1]}"
        format_ = f"Format: {img.format if img.format else 'Unknown'}"
        color = f"Color model: {img.mode}"

        plt.subplot(5, n, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{format_}\n{size}\n{color}", fontsize=10)

        hist = get_histogram(img)

        plt.subplot(5, n, n + i + 1)
        plt.plot(hist, color='black')
        plt.title("Brightness Histogram")
        plt.xlabel("Brightness (0-255)")
        plt.ylabel("Pixel count")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def show_hist(images, title="Filters"):
    if not isinstance(images, list):
        images = [images]

    plt.figure(figsize=(16, 6))

    for i, img in enumerate(images):

        img_equalize_histogram = equalize_histogram(img)
        img_linear_contrast = linear_contrast(img)

        plt.subplot(1, 2, 1)
        plt.imshow(img_equalize_histogram)
        plt.axis('off')
        plt.title("Equalized Histogram")

        plt.subplot(1, 2, 2)
        plt.imshow(img_linear_contrast)
        plt.axis('off')
        plt.title("Linear Contrast")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def show_filters(images, title="Filters"):

    if not isinstance(images, list):
        images = [images]

    plt.figure(figsize=(16, 6))

    for i, img in enumerate(images):

        img_gaussian = gaussian_filter(img)
        img_median = median_filter(img)
        img_box = box_filter(img)
        img_sharp = sharpen_image(img)

        plt.subplot(1, 4, 1)
        plt.imshow(img_gaussian)
        plt.axis('off')
        plt.title("Gaussian Filter")

        plt.subplot(1, 4, 2)
        plt.imshow(img_median)
        plt.axis('off')
        plt.title("Median Filter")

        plt.subplot(1, 4, 3)
        plt.imshow(img_box)
        plt.axis('off')
        plt.title("Box Filter")

        plt.subplot(1, 4, 4)
        plt.imshow(img_sharp)
        plt.axis('off')
        plt.title("Sharpened Image")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def convert_image(img, output_path, output_format):
    img.save(output_path, format=output_format)

def show_neighbors(image, x, y, connectivity=4):
    if isinstance(image, Image.Image):
        image = np.array(image)

    neighbors = get_neighbors(image, x, y, connectivity)

    img_copy = image.copy()
    img_copy[x, y] = [255, 0, 0]
    for nx, ny in neighbors:
        img_copy[nx, ny] = [0, 255, 0]
    
    plt.figure(figsize=(6,6))
    plt.imshow(img_copy)
    plt.axis("off")
    plt.title(f"Піксель ({x}, {y}) та його сусіди ({'4-сусідство' if connectivity == 4 else '8-сусідство'})")
    plt.show()