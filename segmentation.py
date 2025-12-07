from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import os
import numpy as np
from tqdm import tqdm


# take the path of an image, segment, and return masks
def segment_image(image_path, mask_generator): 
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    
    return masks

def segment_images_in_folder(input_folder, output_folder):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # make directory output_folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fnames = os.listdir(input_folder)
    for i in tqdm(range(len(fnames))):
        filename = os.path.join(input_folder, fnames[i])
        masks = segment_image(filename, mask_generator)

        # save masks with same filename but .npy extension
        base_name = os.path.splitext(fnames[i])[0]
        save_path = os.path.join(output_folder, base_name + ".npy")
        np.save(save_path, masks)

if __name__ == "__main__":
    input_folder ="test_images"
    output_folder = "segmentation_masks"
    segment_images_in_folder(input_folder, output_folder)






