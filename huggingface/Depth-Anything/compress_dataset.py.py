import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import glob
import torch.multiprocessing as mp
import random
import re

from tqdm.auto import tqdm


def compress_dataset(images_root, labels_root, output_root):
    image_label_pairs = []
    # Iterate through all the run_ID directories
    for run_id_path in glob.glob(os.path.join(images_root, "run_ID_*")):
        # Extract run ID from the path
        run_id = os.path.basename(run_id_path)
        # For each folder_id within the run_ID directory
        for folder_id_path in glob.glob(os.path.join(run_id_path, "3F", "*")):
            # Extract folder ID from the path
            folder_id = os.path.basename(folder_id_path)
            # For each image within the folder_id directory
            for image_path in glob.glob(os.path.join(folder_id_path, "*.png")):
                # Extract image ID from the path
                image_id = os.path.basename(image_path).split(".")[0]

                segmask_file = image_path.replace("rgb", "segmask")
                output_segmask_file = segmask_file.replace("observation_data_nbv_3F", "compress_observation").replace(".png", ".jpg")
                output_image_path = image_path.replace("observation_data_nbv_3F", "compress_observation").replace(".png", ".jpg")
                segmask_file = segmask_file.replace(".png", ".npy")

                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(output_segmask_file), exist_ok=True)
                cv2.imwrite(
                    output_image_path,
                    cv2.imread(image_path),
                )
                cv2.imwrite(
                    output_segmask_file,
                    np.load(segmask_file, cv2.IMREAD_GRAYSCALE).astype(np.uint8),
                )



images_root = "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/observation_data_nbv_3F"
labels_root = "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/labels_nbv_3F"
output_root = "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/compress_observation"
compress_dataset(images_root, labels_root, output_root)
