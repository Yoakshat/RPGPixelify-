import numpy as np
import os
import cv2
from pixelation import pixelate 
from reduce_colors import reduce_colors

# creating THE pipeline
# segmention, pixelification, color palette reduction

def magical_pixelize(image_name, method, pixel_size): 
    # load masks 
    mask_path = "segmentation_masks/" + os.path.splitext(image_name)[0] + ".npy"
    masks = np.load(mask_path, allow_pickle=True)

    # get the original image
    image = cv2.imread("test_images/" + image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    out_pixelated = np.zeros_like(image)

    for ann in masks:
        mask = ann["segmentation"]
        x, y, w, h = ann["bbox"]

        obj_crop = image[y:y+h, x:x+w]
        mask_crop = mask[y:y+h, x:x+w]

        isolated = np.zeros((h, w, 4), dtype=np.uint8)
        isolated[..., :3] = obj_crop
        isolated[..., 3] = mask_crop.astype(np.uint8) * 255

        px = pixelate(isolated, method=method, pixel_size=pixel_size)
        alpha = mask_crop.astype(bool)

        out_pixelated[y:y+h, x:x+w][alpha]  = px[alpha][:, :3]

    return out_pixelated

# 3 big questions: 
# what are you using to pixelize, what are you using to reduce colors, are you dithering, how many colors?

# pixelize individual, but put it all together and then reduce colors on entire image
# (for consistent theme)
def reduce_entire_image(image_name, px_method, pixel_size, red_method, dither, num_colors): 
    pix = magical_pixelize(image_name, method=px_method, pixel_size=pixel_size)
    # convert back to BGR for reduce_colors
    pix = cv2.cvtColor(pix, cv2.COLOR_RGB2BGR)
    reduced = reduce_colors(pix, method=red_method, dither=dither, k=num_colors)
    return reduced

# pixelize + reduce individual, put it all together
# need less colors (because per individual object)
def reduce_sprites(image_name, px_method, pixel_size, red_method, dither, num_colors): 
    # load masks 
    mask_path = "segmentation_masks/" + os.path.splitext(image_name)[0] + ".npy"
    masks = np.load(mask_path, allow_pickle=True)

    # get the original image
    image = cv2.imread("test_images/" + image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    out_pixelated = np.zeros_like(image)

    for ann in masks:
        mask = ann["segmentation"]
        x, y, w, h = ann["bbox"]

        obj_crop = image[y:y+h, x:x+w]
        mask_crop = mask[y:y+h, x:x+w]

        isolated = np.zeros((h, w, 4), dtype=np.uint8)
        isolated[..., :3] = obj_crop
        isolated[..., 3] = mask_crop.astype(np.uint8) * 255

        px = pixelate(isolated, method=px_method, pixel_size=pixel_size)
        px = cv2.cvtColor(px, cv2.COLOR_RGB2BGR)
        px = reduce_colors(px, method=red_method, dither=dither, k=num_colors)
        alpha = mask_crop.astype(bool)

        out_pixelated[y:y+h, x:x+w][alpha]  = px[alpha][:, :3]

    return out_pixelated

def reduce(image_name, px_method="median", pixel_size=8, red_method="kmeans", dither=True, num_colors=8, per_sprite=False):
    if per_sprite:
        final = reduce_sprites(image_name, px_method, pixel_size, red_method, dither, num_colors=num_colors)
    else:
        final = reduce_entire_image(image_name, px_method, pixel_size, red_method, dither, num_colors=num_colors)
    return final

if __name__ == "__main__":
    # do everything and save it so we can see!
    image_name = "pedestrians.jpg"
    px_method = "median"
    red_method = "kmeans"
    dither = True

    v1 = reduce(image_name, px_method=px_method, pixel_size=8, red_method=red_method, dither=dither, num_colors=3, per_sprite=True)
    v2 = reduce(image_name, px_method=px_method, pixel_size=8, red_method=red_method, dither=dither, num_colors=16, per_sprite=False)

    cv2.imwrite("v1.png", v1)
    cv2.imwrite("v2.png", v2)













    



    
    

