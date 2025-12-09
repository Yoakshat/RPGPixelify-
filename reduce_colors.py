import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from PIL import Image

# return both the centers, image
# all centers are in BGR format
def reduce_kmeans(img, k=8):
    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, n_init="auto")

    labels = kmeans.fit_predict(pixels)
    centers = np.uint8(kmeans.cluster_centers_)

    quantized = centers[labels].reshape(h, w, 3)
    return quantized, kmeans.cluster_centers_

def kmeans_adaptive(img, max_dist=30, k_start=2, k_max=64):
    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3).astype(np.float32)
    k = k_start

    while k <= k_max:
        kmeans = KMeans(n_clusters=k, n_init="auto").fit(pixels)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # compute distance to assigned center
        dists = np.linalg.norm(pixels - centers[labels], axis=1)
        if np.max(dists) <= max_dist:
            break
        k += 1

    quantized = centers[labels].reshape(h, w, 3).astype(np.uint8)
    return quantized, centers

def kmedoids_custom(X, k, max_iter=100):
    N = len(X)
    medoids = np.random.choice(N, k, replace=False)

    for _ in range(max_iter):
        # Compute distances from all points to all medoids
        distances = np.linalg.norm(X[:, None] - X[medoids], axis=2)
        labels = np.argmin(distances, axis=1)

        new_medoids = medoids.copy()

        # Recompute medoids
        for i in range(k):
            cluster_points = np.where(labels == i)[0]
            if len(cluster_points) == 0:
                continue

            # Compute all-pairs distances inside cluster
            cluster_data = X[cluster_points]
            pairwise_dist = np.linalg.norm(
                cluster_data[:, None] - cluster_data, axis=2
            )
            total_dist = np.sum(pairwise_dist, axis=1)

            # Choose the point minimizing total distance
            new_medoids[i] = cluster_points[np.argmin(total_dist)]

        if np.array_equal(new_medoids, medoids):
            break

        medoids = new_medoids

    return medoids, labels

def reduce_kmedoids(img, k=8, sample_size=10000):
    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3).astype(np.float32)

    # Subsample pixels for clustering
    N = len(pixels)
    if N > sample_size:
        sample_idx = np.random.choice(N, sample_size, replace=False)
        X_sample = pixels[sample_idx]
    else:
        X_sample = pixels

    # Run K-Medoids on sample
    medoid_indices_sample, _ = kmedoids_custom(X_sample, k)

    # Convert sample medoid indices back into real colors
    centers = X_sample[medoid_indices_sample].astype(np.uint8)

    # Assign all pixels to nearest medoid
    dists = np.linalg.norm(pixels[:, None] - centers[None, :], axis=2)
    labels = np.argmin(dists, axis=1)

    quantized = centers[labels].reshape(h, w, 3)
    return quantized, centers

def reduce_gmm(img, k=8):
    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3)

    gmm = GaussianMixture(n_components=k, covariance_type="tied")
    gmm.fit(pixels)
    
    labels = gmm.predict(pixels)
    centers = np.uint8(gmm.means_)

    quantized = centers[labels].reshape(h, w, 3)
    return quantized, centers

def reduce_mediancut(img, k=8):
    # img is BGR (cv2), so convert to RGB PIL Image
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    quantized = pil.quantize(colors=k, method=Image.MEDIANCUT)
    # palette is array of R,G,B,R,G,B...
    # get it into numpy array of shape(k, 3)
    palette = np.array(quantized.getpalette()).reshape(-1, 3)
    # switch to BGR
    return cv2.cvtColor(np.array(quantized.convert("RGB")), cv2.COLOR_RGB2BGR), palette[:, ::-1]

# palette is a numpy array of shape (num_colors, 3)
def steinberg_dithering(palette, image): 
    h, w, _ = image.shape
    # normalize to 0-1 so pixel values don't explode
    dithered = image.astype(np.float32)/255.0
    palette = palette.astype(np.float32)/255.0
    
    # process left to right, top to bottom
    for row in range(h): 
        for col in range(w): 
            orig_color = dithered[row][col].copy()
            # find nearest color in palette
            diffs = palette - orig_color
            dists = np.sum(diffs**2, axis=1)
            nearest_color = palette[np.argmin(dists)]
            # set to quantized color
            dithered[row][col] = nearest_color
            
            # diffuse error into nearby pixels if possible
            err_scale = 1.0
            quant_error = (orig_color - nearest_color) * err_scale
            if col + 1 < w:
                dithered[row][col + 1] += quant_error * 7 / 16 
            if col - 1 >= 0 and row + 1 < h:                    
                dithered[row + 1][col - 1] += quant_error * 3 / 16
            if row + 1 < h:
                dithered[row + 1][col] += quant_error * 5 / 16
            if col + 1 < w and row + 1 < h:
                dithered[row + 1][col + 1] += quant_error * 1 / 16

    # cast to uint8
    dithered = np.clip(dithered, 0, 1) * 255.0
    return dithered.astype(np.uint8)

def reduce_colors(img, method="kmeans", dither=True, k=8, max_dist=30):
    """
    Reduce the color palette of an image using the specified method.

    Inputs:
    - img: np.ndarray, BGR image (H, W, 3)
    - method: str, one of "kmeans", "kmedoids", "gmm", "mediancut"
    - k: int, number of colors

    Returns:
    - result: np.ndarray, processed image (same shape as input)
    """

    method = method.lower()

    if method == "kmeans":
        result = reduce_kmeans(img, k)
    elif method == "kmedoids":
        result = reduce_kmedoids(img, k)
    elif method == "gmm":
        result = reduce_gmm(img, k)
    elif method == "mediancut":
        result = reduce_mediancut(img, k)
    elif method == 'kmeans_adaptive':
        result = kmeans_adaptive(img, max_dist)
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced_image, palette = result
    palette = palette.astype(np.uint8)

    if dither: 
        reduced_image = steinberg_dithering(palette, img)

    return reduced_image