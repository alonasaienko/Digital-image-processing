import matplotlib.pyplot as plt
import torchvision # type: ignore
from PIL import Image
import numpy as np

from editing import gaussian_filter, get_histogram, equalize_histogram, median_filter, box_filter, sharpen_image, linear_contrast, get_neighbors, gamma_correction, equalize_histogram_local, add_gaussian_noise, imp_noise, back_filter

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

def show_hist(images, title="Histogram"):
    if not isinstance(images, list):
        images = [images]

    for img in images:
        plt.figure(figsize=(16, 8))

        if img.mode != 'L':
            img_gray = img.convert('L')
        else:
            img_gray = img

        img_global = equalize_histogram(img_gray)
        img_local = equalize_histogram_local(img_gray)
        
        plt.subplot(2, 3, 1)
        plt.imshow(img_gray, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        hist = get_histogram(img_gray)
        plt.plot(hist, color='black')
        plt.title('Original Histogram')
        plt.xlim([0, 256])
        
        plt.subplot(2, 3, 2)
        plt.imshow(img_global, cmap='gray')
        plt.title('Global Equalization')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        hist_global = get_histogram(img_global)
        plt.plot(hist_global, color='black')
        plt.title('Global Equalized Histogram')
        plt.xlim([0, 256])
        
        plt.subplot(2, 3, 3)
        plt.imshow(img_local, cmap='gray')
        plt.title('Local Equalization (CLAHE)')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        hist_local = get_histogram(img_local)
        plt.plot(hist_local, color='black')
        plt.title('Local Equalized Histogram')
        plt.xlim([0, 256])
        
        plt.tight_layout()
        plt.show()

def show_contrast(images, title="Contrast"):
    if not isinstance(images, list):
        images = [images]

    plt.figure(figsize=(18, 12))

    for img in images:

        img_gamma_correction = gamma_correction(img, gamma=0.1)
        img_linear_contrast = linear_contrast(img)

        plt.subplot(3, 2, 1)
        plt.imshow(img, cmap='gray' if img.mode == 'L' else None)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(3, 2, 2)
        hist = get_histogram(img)
        plt.plot(hist, color='black')
        plt.title("Original Histogram")

        plt.subplot(3, 2, 3)
        plt.imshow(img_gamma_correction, cmap='gray' if img_gamma_correction.mode == 'L' else None)
        plt.title("Gamma Correction (gamma=0.1)")
        plt.axis('off')

        plt.subplot(3, 2, 4)
        hist_gamma = get_histogram(img_gamma_correction)
        plt.plot(hist_gamma, color='black')
        plt.title("Gamma Histogram")

        plt.subplot(3, 2, 5)
        plt.imshow(img_linear_contrast, cmap='gray' if img_linear_contrast.mode == 'L' else None)
        plt.title("Linear Contrast")
        plt.axis('off')

        plt.subplot(3, 2, 6)
        hist_linear = get_histogram(img_linear_contrast)
        plt.plot(hist_linear, color='black')
        plt.title("Linear Contrast Histogram")

    plt.tight_layout()
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

def show_filter_results(original_img, filtered_imgs, titles, figsize=(15, 5)):
    """
    Відображення результатів фільтрації
    :param original_img: оригінальне зображення
    :param filtered_imgs: список відфільтрованих зображень
    :param titles: список заголовків
    :param figsize: розмір фігури
    """
    plt.figure(figsize=figsize)
    
    # Відображення оригінального зображення
    plt.subplot(1, len(filtered_imgs)+1, 1)
    if isinstance(original_img, np.ndarray):
        plt.imshow(original_img, cmap='gray' if len(original_img.shape) == 2 else None)
    else:
        plt.imshow(original_img)
    plt.axis('off')
    
    # Відображення відфільтрованих зображень
    for i, (img, title) in enumerate(zip(filtered_imgs, titles), 2):
        plt.subplot(1, len(filtered_imgs)+1, i)
        if isinstance(img, np.ndarray):
            plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        else:
            plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()