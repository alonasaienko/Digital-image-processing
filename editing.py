import random
from fastai.vision.all import *
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2

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

def gamma_correction(image, gamma=1.0):
    if gamma <= 0:
        gamma = 0.01
    
    np_img = np.array(image, dtype=np.float32) / 255.0
    corrected = np.power(np_img, gamma)
    corrected = (corrected * 255).astype(np.uint8)
    return Image.fromarray(corrected)

def equalize_histogram(image):
    return ImageOps.equalize(image) 

def equalize_histogram_local(image, clip_limit=2.0, tile_size=(8,8)):
    np_img = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    equalized = clahe.apply(np_img)
    return Image.fromarray(equalized)

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

def add_gaussian_noise(image, mean=0, sigma=25):
    img_array = np.array(image)
    noise = np.random.normal(mean, sigma, img_array.shape).astype(np.uint8)
    noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)

def imp_noise(image, prob=0.02):
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    noisy_image = np.copy(image)
    height, width = image.shape[0], image.shape[1]
    total_pixels = height * width
    num_pixels = int(prob * total_pixels)

    for _ in range(num_pixels):
        y, x = random.randint(0,height-1), random.randint(0, width-1)
        noisy_image[y, x] = 0 if random.random() < 0.5 else 255
    return noisy_image

def wiener_filter(image, kernel_size=5, K=20):
    img = np.array(image).astype(np.float32)
    
    if img.ndim == 2:
        filtered = _wiener_single_channel(img, kernel_size, K)
    else: 
        channels = []
        for c in range(3):
            filtered_c = _wiener_single_channel(img[:,:,c], kernel_size, K)
            channels.append(filtered_c)
        filtered = np.stack(channels, axis=-1)
    
    return Image.fromarray(np.uint8(np.clip(filtered, 0, 255)))

def _wiener_single_channel(img, kernel_size, K):
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    noise = img - blur
    
    img_fft = np.fft.fft2(img)
    noise_fft = np.fft.fft2(noise)
    
    S_img = np.abs(img_fft)**2
    S_noise = np.abs(noise_fft)**2
    
    H = (S_img - S_noise) / (S_img + K)
    H = np.maximum(H, 0)
    
    filtered_fft = img_fft * H

    filtered = np.fft.ifft2(filtered_fft)
    filtered = np.real(filtered)
    
    return filtered

def back_filter(image, kernel_size=15, K=0.025, noise_sigma=5):
    img = np.array(image.convert('L')).astype(np.float32)
    
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    
    kernel = np.fft.ifftshift(kernel)

    img_fft = np.fft.fft2(img)
    kernel_fft = np.fft.fft2(kernel, s=img.shape)    
    H = np.conj(kernel_fft) / (np.abs(kernel_fft)**2 + K)
    
    restored_fft = H * img_fft
    
    restored = np.fft.ifft2(restored_fft)
    restored = np.abs(restored)
    restored = np.clip(restored, 0, 255).astype(np.uint8)
    
    return Image.fromarray(restored)

def back_filter_rgb(image, kernel_size=15, K=0.025, noise_sigma=5):
    img = np.array(image).astype(np.float32)
    if img.ndim == 2:
        return back_filter(image, kernel_size, K, noise_sigma)

    channels = []
    for c in range(3):
        restored_c = back_filter(Image.fromarray(img[:,:,c].astype(np.uint8)), kernel_size, K, noise_sigma)
        channels.append(np.array(restored_c))
    restored_rgb = np.stack(channels, axis=-1)
    return Image.fromarray(restored_rgb.astype(np.uint8))


def constrained_least_squares(image, kernel_size=5, gamma=0.01):
    img = np.array(image.convert('L'))
    
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian = np.pad(laplacian, ((0, img.shape[0]-3), (0, img.shape[1]-3)), 'constant')

    img_fft = np.fft.fft2(img)
    kernel_fft = np.fft.fft2(kernel, s=img.shape)
    laplacian_fft = np.fft.fft2(laplacian)
    
    kernel_fft_conj = np.conj(kernel_fft)
    restored_fft = (kernel_fft_conj / (np.abs(kernel_fft)**2 + gamma * np.abs(laplacian_fft)**2)) * img_fft
    
    restored = np.fft.ifft2(restored_fft)
    restored = np.abs(restored)
    
    restored = np.uint8(255 * restored / np.max(restored))
    return Image.fromarray(restored)

def constrained_least_squares_rgb(image, kernel_size=5, gamma=0.01):
    channels = image.split()
    restored_channels = [
        constrained_least_squares(ch, kernel_size, gamma) for ch in channels
    ]
    return Image.merge("RGB", restored_channels)
