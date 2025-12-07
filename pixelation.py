import cv2
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def dominant_color(block):
    flat = block.reshape(-1, block.shape[-1])
    return stats.mode(flat, axis=0, keepdims=False)[0]

def safe_resize(img, pixel_size):
    h, w = img.shape[:2]
    new_w = max(1, w // pixel_size)
    new_h = max(1, h // pixel_size)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)


def pixelate_nearest(img, pixel_size=8):
    h, w = img.shape[:2]
    temp = safe_resize(img, pixel_size)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


def pixelate_average(img, pixel_size=8):
    h, w = img.shape[:2]
    out = img.copy()
    for y in range(0, h, pixel_size):
        for x in range(0, w, pixel_size):
            block = img[y:y+pixel_size, x:x+pixel_size]
            if block.size == 0:
                continue
            mean_color = block.reshape(-1, block.shape[-1]).mean(axis=0)
            out[y:y+pixel_size, x:x+pixel_size] = mean_color
    return out


def pixelate_median(img, pixel_size=8):
    h, w = img.shape[:2]
    out = img.copy()
    for y in range(0, h, pixel_size):
        for x in range(0, w, pixel_size):
            block = img[y:y+pixel_size, x:x+pixel_size]
            if block.size == 0:
                continue
            med_color = np.median(block.reshape(-1, block.shape[-1]), axis=0)
            out[y:y+pixel_size, x:x+pixel_size] = med_color
    return out


def pixelate_dominant(img, pixel_size=8):
    h, w = img.shape[:2]
    out = img.copy()
    for y in range(0, h, pixel_size):
        for x in range(0, w, pixel_size):
            block = img[y:y+pixel_size, x:x+pixel_size]
            if block.size == 0:
                continue
            dom = dominant_color(block)
            out[y:y+pixel_size, x:x+pixel_size] = dom
    return out

def pixelate(img, method="average", pixel_size=8): 
    method = method.lower()

    if method == "nearest":       
        return pixelate_nearest(img, pixel_size)
    if method == "average":   
        return pixelate_average(img, pixel_size)
    if method == "median": 
        return pixelate_median(img, pixel_size)
    if method == "dominant":        
        return pixelate_dominant(img, pixel_size)
    else:
        raise ValueError(f"Unknown pixelation method: {method}")

   

