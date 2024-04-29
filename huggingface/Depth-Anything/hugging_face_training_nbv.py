import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

from depth_anything.dpt_nbv import DepthAnything
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

import time

from tqdm.auto import tqdm


class CropTransform:
    def __init__(self, crop_dim):
        self.crop_dim = crop_dim

    def __call__(self, x):
        return x[
            self.crop_dim[0] : self.crop_dim[1], self.crop_dim[2] : self.crop_dim[3]
        ]


class CustomDataset(Dataset):
    def __init__(self, images_root, labels_root):
        self.images_root = images_root
        self.labels_root = labels_root

        self.resize_transform = transforms.Resize(
            (518, 518), interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.image_label_pairs = self._get_image_label_pairs()

        self.pattern = r"nbv_(\d+)_"
        self.pattern_rgb = r"rgb_(\d+)_"

    def _get_image_label_pairs(self):
        pairs = []
        # Iterate through all the run_ID directories
        for run_id_path in glob.glob(os.path.join(self.images_root, "run_ID_*")):
            # Extract run ID from the path
            run_id = os.path.basename(run_id_path)
            # For each folder_id within the run_ID directory
            for folder_id_path in glob.glob(os.path.join(run_id_path, "3F", "*")):
                folder_id = os.path.basename(folder_id_path)
                # For each image file within the folder_id directory
                for image_file in glob.glob(
                    os.path.join(folder_id_path, "rgb_*_7_*.png")
                ):
                    # Derive the base name for the image file to match labels
                    base_name = os.path.splitext(os.path.basename(image_file))[0]
                    count = base_name.split("_")[-1]

                    # Pattern for corresponding label files
                    nbv_label_pattern = os.path.join(
                        self.labels_root,
                        run_id,
                        "3F",
                        folder_id,
                        f"nbv_*_{folder_id}_{count}.npz",
                    )

                    # Find all matching label files and add to pairs
                    segmask_file = image_file.replace("rgb", "segmask")
                    for label_file in glob.glob(nbv_label_pattern):
                        # label_value = np.load(label_file)["arr_0"]
                        pairs.append((image_file, label_file, segmask_file))
        return pairs

    def __getitem__(self, idx):
        image_file, label_file, segmask_file = self.image_label_pairs[idx]
        # Load image
        numpy_image = cv2.imread(image_file).astype(np.uint8)
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        crop_bin_dim = [73, 287, 195, 442]

        numpy_image = CropTransform(crop_bin_dim)(numpy_image)
        numpy_image = np.array(numpy_image).astype(np.float32) / 255.0

        image = torch.from_numpy(numpy_image).permute(2, 0, 1).float().div(255)
        image = self.normalize_transform(image)
        image = self.resize_transform(image)

        # Load label
        match = re.search(self.pattern_rgb, image_file).group(1)
        label = np.load(label_file)["arr_0"].astype(np.float32)

        # Load mask
        numpy_mask = cv2.imread(segmask_file, cv2.IMREAD_GRAYSCALE)
        match = re.search(self.pattern, label_file)
        mask_id = match.group(1)

        numpy_mask = CropTransform(crop_bin_dim)(numpy_mask)
        mask = numpy_mask == int(mask_id)

        mask = torch.from_numpy(mask).unsqueeze(0)
        mask = self.resize_transform(mask)

        return image, label, mask

    def __len__(self):
        return len(self.image_label_pairs)


def init_model():
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")
    model = model.to("cuda")
    # model.load_state_dict(
    #     torch.load(
    #         "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/Depth-Anything/huggingface/Depth-Anything/temp.pth"
    #     )
    # )

    for name, param in model.named_parameters():
        if "pretrained" in name or "depth_head.scratch.output_conv" in name:
            param.requires_grad = False
        # print(name, param.requires_grad)

    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-1
    # )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    return model, criterion, optimizer, scheduler


def train(train_dataloader, val_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epoch = 50

    model, criterion, optimizer, scheduler = init_model()

    # progress_bar = tqdm(range(num_epoch*len(train_dataloader) + num_epoch*len(val_dataloader)))

    running_loss = 0.0

    scaling_factor = 10
    gamma = 2

    model.train()

    lowest_loss = 10000.0

    for epoch in range(num_epoch):
        running_loss = 0.0

        for images, labels, mask in train_dataloader:
            images, labels, mask = images.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()

            depth_out, nbv_out = model(images, mask)
            loss = criterion(nbv_out, labels)
            scaled_loss = loss * scaling_factor

            scaled_loss.backward()
            optimizer.step()
            running_loss += scaled_loss.item()

            # progress_bar.update(1)
            # print(f"Loss: {loss.item()}")

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

        model.eval()  # Set the model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        validation_loss = 0.0

        with torch.no_grad():  # Disable gradient computation
            for images, labels, mask in val_dataloader:
                images, labels, mask = (
                    images.to(device),
                    labels.to(device),
                    mask.to(device),
                )

                depth_out, nbv_out = model(images, mask)
                loss = criterion(nbv_out, labels)
                scaled_loss = loss * scaling_factor

                validation_loss += scaled_loss.item()

                # progress_bar.update(1)
        avg_validation_loss = validation_loss / len(val_dataloader)
        if avg_validation_loss < lowest_loss:
            lowest_loss = avg_validation_loss
            torch.save(
                model,
                "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/Depth-Anything/huggingface/Depth-Anything/checkpoint_finetune_nbv.pth",
            )
        print(f"Validation Loss: {avg_validation_loss}")

        model.train()

        scheduler.step(avg_validation_loss)


if __name__ == "__main__":
    # Set the start method to 'spawn'
    mp.set_start_method("spawn", force=True)

    dataset = CustomDataset(
        images_root="/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/compress_observation_2",
        labels_root="/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/labels_nbv_3F_2",
    )

    print(len(dataset))

    # Determine sizes for training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=16
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=16
    )

    # Example training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(train_dataloader, val_dataloader)
