# 🎉 Gaussian Splatting Lab: 3D Foundation Model + gsplat

Welcome to the **Gaussian Splatting Lab**! 🚀 This repository provides an academic playground to experiment with Gaussian Splatting for photorealistic 3D scene reconstruction. We're constantly improving—feel free to spot issues or contribute! 💡

---

## 🌟 Features

- 🗺 **Dense Point Cloud Estimation** via the VGGT 3D Foundation Model
- 🎨 **2D Gaussian Splatting** training with **gsplat**
- 🌐 **Web-based Viewer** for real-time visualization
- 🎥 **Video Rendering** of your scene’s trajectory
- 📊 **Appearance Evaluations** to assess photorealism

> 🔍 **Tip:** Tweak any hyperparameter by inspecting `base_config.py`.

---

## 📥 Installation

1. **Clone with Submodules** (VGGT):
   ```bash
   git clone <repo-url>
   cd gs_lab 
   git submodule update --init --recursive
   ```

2. **Install Requirements** (Python + CUDA 11.7):
   ```bash
   pip install -r requirements.txt
   ```


---

## 🎬 Demo

#### Watch our quick demo on YouTube:

<iframe width="640" height="360" src="https://www.youtube.com/embed/n_6iYyrO3x8" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


---

## 🔨 Usage

### 1. Keyframe Selection (Optional)
If you have **>200 images**, select a subset to avoid OOM errors:
```bash
python keyframe_selection \
  --input_folder path/to/images \
  --num_keyframes 150 \
  --output_folder path/to/keyframes
```

**Folder structure** after this step:
```
data/
└── dataset/
    └── scene/
        └── images/
```

---

### 2. Dense Point Cloud Estimation

1. **Download VGGT checkpoint** from [Hugging Face](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt).
2. **Run inference**:
   ```bash
   python pc_inference \
     --model_path path/to/model.pt \
     --img_base_path data/dataset/scene \
     --init_conf_threshold 0.5 \
     --vis_point_cloud \
     --downsample_factor 2 \
     --save_depths
   ```
3. **Assets generated**:
   ```
   data/
   └── dataset/
       └── scene/
           ├── images/
           ├── depths/      # Dense depth maps
           └── sparse/0/    # COLMAP-like outputs
   ```

> ⚙️ The `--vis_point_cloud` flag launches a **Viser** web viewer.

---

### 3. Gaussian Splatting Training

Train your scene with **gsplat**:
```bash
python trainer_2dgs.py \
  --data_dir data/dataset/scene \
  --result_dir results/dataset/scene
```

---

### 4. Visualization & Rendering

- **Interactive Viewer**:
  ```bash
  python gaussian_splatting/simple_viewer.py \
    --ckpt results/dataset/scene/checkpoint.pth
  ```

- **Render a Trajectory Video**:
  ```bash
  python render_trajectory.py \
    --ckpt results/dataset/scene/checkpoint.pth \
    --out_video results/dataset/scene/demo.mp4
  ```

---

## 🤝 Contributing

- 🐛 **Report issues** or suggest features on GitHub.
- 📥 **Submit pull requests**—we welcome all improvements!

---

## 🙏 Acknowledgments

We gratefully acknowledge [gsplat by Nerfstudio](https://github.com/nerfstudio-project/gsplat) and [VGGT by Facebook Research](https://github.com/facebookresearch/vggt) for their fantastic frameworks that power this project.