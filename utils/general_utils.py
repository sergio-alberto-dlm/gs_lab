import os 
import numpy as np 
import cv2 

def save_depth_maps(depth_maps, output_folder, images_names):
    """
    Save a list of 2D NumPy arrays (depth maps) into `output_folder` as PNGs.
    """
    os.makedirs(output_folder, exist_ok=True)
    for i, dm in enumerate(depth_maps):
        # # ensure 2D
        # if dm.ndim != 2:
        #     raise ValueError(f"Depth map at index {i} is not 2D (shape={dm.shape})")
        # re-scaled 
        arr = (dm - dm.min()) / (dm.max() - dm.min())
        arr = ((arr * 255).astype(np.uint16))

        filename = images_names[i]
        out_path = os.path.join(output_folder, filename)
        # write 16-bit PNG
        if not cv2.imwrite(out_path, arr):
            raise IOError(f"Failed to write {out_path}")

    return 