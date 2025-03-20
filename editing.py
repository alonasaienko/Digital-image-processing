from fastai.vision.all import *
import numpy as np
from PIL import Image, ImageOps, ImageFilter

def gaussian_filter(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))

def median_filter(image, size=3):
    return image.filter(ImageFilter.MedianFilter(size))

def box_filter(image, size=3):
    return image.filter(ImageFilter.BoxBlur(size))

def sharpen_image(image, factor=2):
    return image.filter(ImageFilter.UnsharpMask(radius=3, percent=250, threshold=3))

def linear_contrast(image):
    gray = image.convert('L')
    np_img = np.array(gray)

    min_val, max_val = np.min(np_img), np.max(np_img)
    stretched = (np_img - min_val) * (255 / (max_val - min_val))
    stretched_img = Image.fromarray(stretched.astype(np.uint8))

    return stretched_img

def equalize_histogram(image):
    return ImageOps.equalize(image) 

def get_histogram(image):
    gray_img = image.convert('L')
    hist = gray_img.histogram()
    return hist

def get_neighbors(image, x, y, connectivity=4):
    neighbors = []

    if isinstance(image, Image.Image):
        image = np.array(image)
    
    rows, cols = image.shape[:2]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    if connectivity == 8:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0<= nx < rows and 0 <= ny < cols:
            neighbors.append((nx, ny))
    return neighbors

def quantize_image(image, color_depth=2):
    img_array = np.array(image)

    step = 256 // color_depth
    img_quantized = (img_array // step) * step
    
    quantized_image = Image.fromarray(img_quantized.astype('uint8'))
    
    return quantized_image