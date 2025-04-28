# 🚧 🚧 🚧 

# Gaussian Splatting Lab: 3D Foundation Model + gsplat

The following repository arises as an academic tool to play and experiment with gaussian splatting. The project is still under construction, please do not hesitate to inform any mistakes and contributions are welcome. 

This project is mainly build on top of the recent 3D Foundation Model VGGT and gsplat which is a specialized framework to train gaussian splatting scenes. First with VGGT we estimate an initial dense point cloud which serves as input to the training pipeline of gaussian splatting. You just need a set of images of the same scene and by following the instructions below you will obtain a photorealistic 3D model, GPU hardware is required! Currently within this repo you are able to:

1. Estimate a dense point cloud. 
2. Train a 2D gaussian splatting scene. 
3. Web-based viewer. 
4. Render a video to visualize results.
5. Perform appearence evaluations.

We higly motivates you to check the base_config.py file to modify any hyperparameter of the training pipeline 

### installation 

First of all clone this repo along with the submodules (VGGT)

```bash 
git clone <repo-url>
cd <your-main-repo-folder>
git submodule update --init --recursive
````

Then install the requirements. We use pytorch with cuda version 11.7

```bash 
pip install -r requirements.txt 
```

### demo 
<video controls width="640">
  <source src="assets/traj_29999.mp4" type="video/mp4">
  <!-- fallback link if video tag isn’t supported: -->
  <p>Your browser doesn’t support HTML5 video. 
     <a href="assets/traj_29999.mp4">Download the video instead.</a>
  </p>
</video>

### dense point cloud estimation 

If the number of images is greater than 200 we highly recommend to first select some keyframes less than this number, because you might get out of memory for VGGT inference. Run:

```bash 
python keyframe_selection --input_folder input_folder/path --num_keyframes 150 --output_folder output_folder/path
```

The output folder path should have the following structure 

```bash 
data
    |_ dataset
    |   |_ scene
    |   |   |_ images
```

To run VGGT first download the checkppoint [here](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt). Now we can estimate a dense point cloud with. Run:

```bash 
python pc_inference --model_path --img_base_path data/dataset/scene --init_conf_threshold --vis_point_cloud --downsample_factor --save_depths
``` 

The command --vis_point_cloud will lunch a web-based viewer with viser. Now you should have the following data structure

```bash 
data
    |_ dataset
    |   |_ scene
    |   |   |_ images
    |   |   |_ depths
    |   |   |_ sparse/0
```

Within depths youo can find the depth map associated to the images and in sparse/0 you can find the colmap-like data 

### gsplat training 

Finally you can lunch a gaussian splatting training with gsplat. Run:

```bash 
python trainer_2dgs.py --data_dir data/dataset/scene --result_dir results/dataset/scene 
``` 

### visualizatin 

To visualise your scene run: 

```bash 
python gaussian_splatting/simple_viewer.py --ckpt /path/to/checkpoint 
```

The path to the checkpoint should be at: 