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
    reduced = reduce_colors(pix, method=red_method, dither=dither, k=num_colors)

    # get the reduced sprites (e.g. go back to masks)
    mask_path = "segmentation_masks/" + os.path.splitext(image_name)[0] + ".npy"
    masks = np.load(mask_path, allow_pickle=True)
    reduced_sprites = [] 
    for ann in masks: 
        mask = ann["segmentation"]
        x, y, w, h = ann["bbox"]

        reduced_sprite = np.zeros((h, w, 3))
        alpha = mask[y:y+h, x:x+w].astype(bool)
        reduced_sprite[alpha] = reduced[y:y+h, x:x+w][alpha]

        reduced_sprites.append(reduced_sprite)

    return reduced, reduced_sprites

# pixelize + reduce individual, put it all together
# need less colors (because per individual object)
def reduce_sprites(image_name, px_method, pixel_size, red_method, dither, num_colors): 
    # load masks 
    mask_path = "segmentation_masks/" + os.path.splitext(image_name)[0] + ".npy"
    masks = np.load(mask_path, allow_pickle=True)

    # get the original image
    image = cv2.imread("test_images/" + image_name)  # already BGR
    out_pixelated = np.zeros_like(image)
    reduced_sprites = [] 

    for ann in masks:
        mask = ann["segmentation"]
        x, y, w, h = ann["bbox"]

        obj_crop = image[y:y+h, x:x+w]  # still BGR
        mask_crop = mask[y:y+h, x:x+w]

        isolated = np.zeros((h, w, 4), dtype=np.uint8)
        isolated[..., :3] = obj_crop
        isolated[..., 3] = mask_crop.astype(np.uint8) * 255

        alpha = mask_crop.astype(bool)
        # pixelate first, then reduce colors
        px = pixelate(isolated, method=px_method, pixel_size=pixel_size)
        px_bgr = reduce_colors(px[..., :3], method=red_method, dither=dither, k=num_colors)

        # apply the mask to the output
        out_pixelated[y:y+h, x:x+w][alpha] = px_bgr[alpha]

        red_sprite = np.zeros((h, w, 3), dtype=np.uint8)
        red_sprite[alpha] = px_bgr[alpha]
        reduced_sprites.append(red_sprite)

    return out_pixelated, reduced_sprites

def reduce(image_name, px_method="median", pixel_size=8, red_method="kmeans", dither=True, num_colors=8, per_sprite=False):
    if per_sprite:
        final, indv_sprites = reduce_sprites(image_name, px_method, pixel_size, red_method, dither, num_colors=num_colors)
    else:
        final, indv_sprites = reduce_entire_image(image_name, px_method, pixel_size, red_method, dither, num_colors=num_colors)
    return final, indv_sprites

if __name__ == "__main__":
    # do everything and save it so we can see!
    image_name = "japancity.jpg"
    px_method = "median"
    red_method = "kmeans"
    dither = True

    v1, indv_sprites = reduce(image_name, px_method=px_method, pixel_size=8, red_method=red_method, dither=dither, num_colors=3, per_sprite=True)
    # make a folder to save individual sprites for each image
    sprite_folder = "output_sprites/" + os.path.splitext(image_name)[0]
    if not os.path.exists(sprite_folder):
        os.makedirs(sprite_folder)

    for i, sprite in enumerate(indv_sprites):
        # only save reasonably sized sprites
        if sprite.shape[0] >= 80 or sprite.shape[1] >= 80: 
            cv2.imwrite(os.path.join(sprite_folder, f"sprite_{i}.png"), sprite)

    cv2.imwrite("pixelated_" + image_name, v1)













    



    
    

