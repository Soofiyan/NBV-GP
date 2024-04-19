# Born out of Issue 36.
# Allows  the user to set up own test files to infer on (Create a folder my_test and add subfolder input and output in the metric_depth directory before running this script.)
# Make sure you have the necessary libraries
# Code by @1ssb

import argparse
import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import open3d as o3d
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import cv2
import matplotlib.pyplot as plt

# Global settings
FL = 715.0873
FY = 256 * 0.6
FX = 256 * 0.6
NYU_DATA = False
FINAL_HEIGHT = 256
FINAL_WIDTH = 256
INPUT_DIR = "./my_test/input"
OUTPUT_DIR = "/tmp/"
DATASET = "nyu"  # Lets not pick a fight with the model's dataloader


def process_images(model):

    image_path = "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/observation_data_nbv_3F/run_ID_0/3F/0/rgb_0_1.npy"

    numpy_image = np.load(image_path)
    color_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    color_image = Image.fromarray(color_image)

    original_width, original_height = color_image.size
    image_tensor = (
        transforms.ToTensor()(color_image)
        .unsqueeze(0)
        .to("cuda" if torch.cuda.is_available() else "cpu")
    )

    pred = model(image_tensor, dataset=DATASET)
    if isinstance(pred, dict):
        pred = pred.get("metric_depth", pred.get("out"))
    elif isinstance(pred, (list, tuple)):
        pred = pred[-1]
    pred = pred.squeeze().detach().cpu().numpy()

    # Resize color image and depth to final size
    resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)
    resized_pred = Image.fromarray(pred).resize(
        (FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST
    )

    focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
    x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
    x = (x - FINAL_WIDTH / 2) / focal_length_x
    y = (y - FINAL_HEIGHT / 2) / focal_length_y
    z = np.array(resized_pred)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = np.array(resized_color_image).reshape(-1, 3) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries(
        [pcd],
        zoom=0.3412,
        front=[0.4257, -0.2125, -0.8795],
        lookat=[2.6172, 2.0475, 1.532],
        up=[-0.0694, -0.9768, 0.2024],
    )
    # o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(image_path))[0] + ".ply"), pcd)


def main(model_name, pretrained_resource):
    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    process_images(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="zoedepth", help="Name of the model to test"
    )
    parser.add_argument(
        "-p",
        "--pretrained_resource",
        type=str,
        default="local::/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/Depth-Anything/metric_depth/checkpoints/depth_anything_metric_depth_indoor.pt",
        help="Pretrained resource to use for fetching weights.",
    )

    args = parser.parse_args()
    main(args.model, args.pretrained_resource)
