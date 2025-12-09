import cv2
import numpy as np
import os

import cv2
import numpy as np
import os

import cv2
import numpy as np
import os

def save_sprite_grid(folder, out_path="sprite_grid.png"):
    indices = [0, 1, 2, 3, 4, 5]
    sprites = []
    labels = []

    for i in indices:
        path = os.path.join(folder, f"sprite_{i}.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Could not load: {path}")

        # --- Transparent BG → white ---
        if img.shape[2] == 4:
            rgb = img[..., :3]
            a = img[..., 3:4] / 255.0
            white = np.ones_like(rgb) * 255
            img = (rgb * a + white * (1 - a)).astype(np.uint8)

        # --- No alpha? Remove black bg anyway ---
        if img.shape[2] == 3:
            mask = np.all(img < 10, axis=2)  # near black
            img = img.copy()
            img[mask] = [255, 255, 255]

        sprites.append(img)
        labels.append(f"Sprite {i}")

    max_h = max(s.shape[0] for s in sprites)
    max_w = max(s.shape[1] for s in sprites)
    label_space = 40
    tile_h = max_h + label_space
    tile_w = max_w

    grid = np.ones((2*tile_h, 3*tile_w, 3), dtype=np.uint8) * 255

    for idx, (sprite, label) in enumerate(zip(sprites, labels)):
        r = idx // 3
        c = idx % 3

        tile_y = r * tile_h
        tile_x = c * tile_w

        h, w = sprite.shape[:2]
        y0 = tile_y + label_space + (max_h - h)//2
        x0 = tile_x + (max_w - w)//2

        grid[y0:y0+h, x0:x0+w] = sprite

        cv2.putText(grid, label,
                    (tile_x + 5, tile_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,0,0), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, grid)
    print(f"Saved sprite grid → {out_path}")



if __name__ == "__main__":
    save_sprite_grid(folder="output_sprites/japancity",out_path="palette/pixelated_sprite_mediancut4.png")
