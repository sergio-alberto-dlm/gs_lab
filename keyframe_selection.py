#!/usr/bin/env python3
"""
select_keyframes.py

Selects N keyframes from a folder of images by clustering
HSV color histograms and picking the medoid of each cluster.
Copies selected frames into an output folder.
"""

import os
import argparse
import cv2
import numpy as np
import shutil


def compute_hist(path, h_bins=30, s_bins=32, resize_dim=(320,240)):
    """
    Read image, resize for speed, convert to HSV, compute 
    2D histogram over H and S, normalize, and flatten.
    """
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    img = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [h_bins, s_bins], [0,180, 0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def select_keyframes(image_paths, num_keyframes):
    """
    Cluster per-frame histograms into num_keyframes groups,
    then pick one representative per cluster.
    """
    # build data matrix
    hists = [compute_hist(p) for p in image_paths]
    data = np.array(hists, dtype=np.float32)

    # cluster with OpenCV kmeans
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_COUNT, 100, 0.2)
    flags = cv2.KMEANS_PP_CENTERS
    _, labels, centers = cv2.kmeans(
        data, num_keyframes, None, criteria, 10, flags
    )

    # pick medoid of each cluster
    keyframe_idxs = []
    for cid in range(num_keyframes):
        idxs = np.where(labels.flatten() == cid)[0]
        dists = np.linalg.norm(data[idxs] - centers[cid], axis=1)
        best = idxs[np.argmin(dists)]
        keyframe_idxs.append(best)

    keyframe_idxs.sort()
    return [image_paths[i] for i in keyframe_idxs]


def main():
    parser = argparse.ArgumentParser(
        description="Select keyframes by HSV-histogram clustering and copy them"
    )

    parser.add_argument("--input_folder", required=True,
                        help="Folder containing your images (e.g. extracted frames)")
    parser.add_argument("--num_keyframes", type=int, required=True,
                        help="How many keyframes to select")
    parser.add_argument("--output_folder", required=True,
                        help="Folder in which to copy selected keyframes")
    args = parser.parse_args()

    # ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # gather and sort image paths
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_files = sorted(os.listdir(args.input_folder))
    image_paths = [
        os.path.join(args.input_folder, f)
        for f in all_files
        if os.path.splitext(f.lower())[1] in exts
    ]

    if len(image_paths) < args.num_keyframes:
        raise ValueError(
            f"Only {len(image_paths)} images found, but {args.num_keyframes} requested."
        )

    keyframes = select_keyframes(image_paths, args.num_keyframes)

    # copy each selected keyframe to the output folder
    for src_path in keyframes:
        fname = os.path.basename(src_path)
        dst_path = os.path.join(args.output_folder, fname)
        shutil.copy(src_path, dst_path)

    print(f"✔ {len(keyframes)} keyframes copied to '{args.output_folder}'")


if __name__ == "__main__":
    main()