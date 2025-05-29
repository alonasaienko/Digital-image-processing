import random
from fastai.vision.all import *
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
from sklearn.cluster import KMeans

def gaussian_filter(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))

def median_filter(image, size=3):
    return image.filter(ImageFilter.MedianFilter(size))

def box_filter(image, size=3):
    return image.filter(ImageFilter.BoxBlur(size))

def sharpen_image(image, factor=2):
    return image.filter(ImageFilter.UnsharpMask(radius=3, percent=250, threshold=3))

def sharpen_image_kernel(image):
    sharpen_kernel = [
        [0, -1, 0],
        [-1, 5,-1],
        [0, -1, 0]
    ]
    return image.filter(ImageFilter.Kernel(size=(3, 3), kernel=[item for sublist in sharpen_kernel for item in sublist], scale=1))

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

def rgb_to_hsv(image):
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert('RGB'))
    else:
        rgb = np.array(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return Image.fromarray(hsv, 'RGB')

def rgb_to_cmy(image):
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
    else:
        rgb = np.array(image, dtype=np.float32) / 255.0
    
    cmy = 1.0 - rgb
    cmy = (cmy * 255).astype(np.uint8)
    return Image.fromarray(cmy)

def rgb_to_ycbcr(image):
    if isinstance(image, Image.Image):
        ycbcr = image.convert('YCbCr')
    else:
        ycbcr = Image.fromarray(image).convert('YCbCr')
    return ycbcr

def segmentation(image):
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert('RGB'))
    else:
        rgb = np.array(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (30, 50, 50), (90, 255, 255))
    segmented = cv2.bitwise_and(rgb, rgb, mask=mask)
    return Image.fromarray(segmented, 'RGB')

def kmeans_segmentation(image, k=3):
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert('RGB'))
    else:
        rgb = np.array(image)

    Z = rgb.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(Z)
    labels = kmeans.labels_
    centers = np.uint8(kmeans.cluster_centers_)

    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(rgb.shape)

    return Image.fromarray(segmented_image)

def color_blur(image, kernel_size=(5, 5), sigma=1):
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert('RGB'))
    else:
        rgb = np.array(image)
    (b, g, r) = cv2.split(rgb)

    b_blur = cv2.GaussianBlur(b, kernel_size, sigma)
    g_blur = cv2.GaussianBlur(g, kernel_size, sigma)
    r_blur = cv2.GaussianBlur(r, kernel_size, sigma)

    return cv2.merge((b_blur, g_blur, r_blur))

def compress_image(input_image_path, output_image_path, quality=85):
    image = Image.open(input_image_path)
    
    image.save(output_image_path, 'JPEG', quality=quality)

def erosion(image, kernel_size=3, kernel_shape=cv2.MORPH_RECT):
    if isinstance(image, Image.Image):
        img = np.array(image.convert('RGB'))
    else:
        img = np.array(image)
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return cv2.erode(img, kernel, iterations=1)

def dilation(image, kernel_size=3, kernel_shape=cv2.MORPH_RECT):
    if isinstance(image, Image.Image):
        img = np.array(image.convert('RGB'))
    else:
        img = np.array(image)
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return cv2.dilate(img, kernel, iterations=1)

def opening(image, kernel_size=3, kernel_shape=cv2.MORPH_RECT):
    if isinstance(image, Image.Image):
        img = np.array(image.convert('RGB'))
    else:
        img = np.array(image)
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closing(image, kernel_size=3, kernel_shape=cv2.MORPH_RECT):
    if isinstance(image, Image.Image):
        img = np.array(image.convert('RGB'))
    else:
        img = np.array(image)
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def gray_erosion(image, kernel_size=3, kernel_shape=cv2.MORPH_RECT):
    if isinstance(image, Image.Image):
        img = np.array(img.convert('L'))
    else:
        img = np.array(image)
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return cv2.erode(img, kernel)

def gray_dilation(image, kernel_size=3, kernel_shape=cv2.MORPH_RECT):
    if isinstance(image, Image.Image):
        img = np.array(img.convert('L'))
    else:
        img = np.array(image)
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return cv2.dilate(img, kernel)

def gray_opening(image, kernel_size=3, kernel_shape=cv2.MORPH_RECT):
    if isinstance(image, Image.Image):
        img = np.array(img.convert('L'))
    else:
        img = np.array(image)
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def gray_closing(image, kernel_size=3, kernel_shape=cv2.MORPH_RECT):
    if isinstance(image, Image.Image):
        img = np.array(img.convert('L'))
    else:
        img = np.array(image)
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def dilation_reconstruction(marker, mask, kernel_size=3, max_iter=100):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    prev = np.zeros_like(marker)
    reconstruction = marker.copy()
    
    for _ in range(max_iter):
        reconstruction = cv2.dilate(reconstruction, kernel)
        reconstruction = cv2.bitwise_and(reconstruction, mask)
        
        if np.array_equal(reconstruction, prev):
            break
        prev = reconstruction.copy()
    
    return reconstruction

def erosion_reconstruction(marker, mask, kernel_size=3, max_iter=100):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    prev = np.zeros_like(marker)
    reconstruction = marker.copy()
    
    for _ in range(max_iter):
        reconstruction = cv2.erode(reconstruction, kernel)
        reconstruction = cv2.bitwise_or(reconstruction, mask)
        
        if np.array_equal(reconstruction, prev):
            break
        prev = reconstruction.copy()
    
    return reconstruction