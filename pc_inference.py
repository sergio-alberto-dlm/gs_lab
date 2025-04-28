import sys, os, glob 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "submodules", "vggt")))

import argparse
import torch
import open3d as o3d
import numpy as np 
from PIL import Image

from submodules.vggt.vggt.models.vggt import VGGT
from submodules.vggt.vggt.utils.load_fn import load_and_preprocess_images
from submodules.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from submodules.vggt.vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3
from utils.geometry_utils import viz_point_cloud, save_colmap_cameras, save_colmap_images, downsample_pcd, storePly
from utils.general_utils import save_depth_maps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/sergio/checkpoints/vggt/model.pt", help="Path to model weights")
    parser.add_argument("--img_base_path", type=str, required=True, help="Base path to images")
    parser.add_argument("--use_point_map", action="store_true", help="Flag to use point map otherwise recover the point cloud from depth maps")
    parser.add_argument("--init_conf_threshold", type=int, default=50, help="Confidence threshold for cleaning point cloud")
    parser.add_argument("--vis_point_cloud", action="store_true", help="Flag to visualize point cloud")
    parser.add_argument("--downsample_factor", type=float, default=4, help="Random downsample factor to the point cloud")
    parser.add_argument("--save_depths", action="store_true", default=True, help="Flag to save depth maps into colmap dir")
    return parser

def load_images(path_images:str):
    """load images from a directory"""
    images_names = sorted(os.listdir(path_images))
    images_paths = [os.path.join(path_images, name) for name in images_names]

    print("num images: ", len(images_paths))
    # retrieve the original image size 
    image = Image.open(images_paths[0])
    width, height = image.size

    # preproceess images 
    images = load_and_preprocess_images(images_paths).to(DEVICE)

    return images, (width, height), images_names 

def load_model(model_path:str):
    """load the pre-trained model""" 
    print("Loading model...")
    # Initialize model
    model = VGGT()
    # Load the state dict
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    model = model.to(device=DEVICE)
    return model 

def pc_inference(model, images, use_point_map, init_conf_threshold=50):
    """perform point cloud inference"""
    print("Running VGGT inference...")    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsics_cam, intrinsics_cam = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    extrinsics_cam = extrinsics_cam.cpu().numpy().squeeze(0)
    intrinsics_cam = intrinsics_cam.cpu().numpy().squeeze(0)

    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

    # Unpack prediction dict
    images = predictions["images"]  # (S, 3, H, W)
    world_points_map = predictions["world_points"]  # (S, H, W, 3)
    conf_map = predictions["world_points_conf"]  # (S, H, W)

    depth_map = predictions["depth"]  # (S, H, W, 1)
    depth_conf = predictions["depth_conf"]  # (S, H, W)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world_mat[:, :3, -1] -= scene_center

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    points=points_centered[init_conf_mask]
    colors=colors_flat[init_conf_mask]

    return {
        "extrinsics": cam_to_world_mat, 
        "intrinsics": intrinsics_cam, 
        "depth_maps": depth_map, 
        "points": points, 
        "colors": colors,
        "confs": conf
    }

def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # directories 
    img_folder_path = os.path.join(args.img_base_path, "images")
    depth_folder_path = os.path.join(args.img_base_path, "depths")
    output_colmap_path = os.path.join(args.img_base_path, "sparse/0")
    os.makedirs(output_colmap_path, exist_ok=True)
    os.makedirs(depth_folder_path, exist_ok=True)

    # model inference 
    images, ori_size, images_names = load_images(path_images=img_folder_path)
    model = load_model(model_path=args.model_path)
    predictions = pc_inference(
        model=model, images=images, use_point_map=args.use_point_map, init_conf_threshold=args.init_conf_threshold
    )

    # unpack predictions 
    points = predictions["points"]
    colors = predictions["colors"]
    extrinsics = predictions["extrinsics"]
    intrinsics = predictions["intrinsics"]

    # debug/visualize point cloud 
    if args.vis_point_cloud: viz_point_cloud(points=points, colors=colors)

    # save colmap outputs
    save_colmap_cameras(ori_size, intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(extrinsics, os.path.join(output_colmap_path, 'images.txt'), sorted(images_names))

    # save final point cloud
    filtered_points, filtered_colors = downsample_pcd(points=points, colors=colors, downsample_factor=args.downsample_factor)
    print("number of points: ", filtered_points.shape[0])
    storePly(os.path.join(output_colmap_path, "points3D.ply"), filtered_points, filtered_colors)

    # save depth maps 
    if args.save_depths: save_depth_maps(depth_maps=predictions["depth_maps"], output_folder=depth_folder_path, images_names=images_names)

    print("Geometry estimation completed.")

if __name__ == "__main__":
    main()
