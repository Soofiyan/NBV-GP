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
            (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.augmentation_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
            ]
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
                    os.path.join(folder_id_path, "rgb_*_7_*.jpg")
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
                    segmask_file = image_file.replace("rgb", "segmask").replace("jpg", "png")
                    for label_file in glob.glob(nbv_label_pattern):
                        label = np.load(label_file)["arr_0"].astype(np.float32)
                        if label == 2 and random.random() < 0.5:
                            continue
                        pairs.append((image_file, label_file, segmask_file))
                        pairs.append((image_file, label_file, segmask_file))
        return pairs

    def __getitem__(self, idx):
        image_file, label_file, segmask_file = self.image_label_pairs[idx]
        # Load image
        numpy_image = cv2.imread(image_file).astype(np.uint8)
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        crop_bin_dim = [73, 287, 195, 442]

        numpy_image = CropTransform(crop_bin_dim)(numpy_image)

        # plt.imshow(numpy_image)
        # plt.show()

        # Load mask
        numpy_mask = cv2.imread(segmask_file, cv2.IMREAD_GRAYSCALE)
        match = re.search(self.pattern, label_file)
        mask_id = match.group(1)
        numpy_mask = CropTransform(crop_bin_dim)(numpy_mask)
        mask = numpy_mask == int(mask_id)

        # plt.imshow(mask)
        # plt.show()

        # w = 10
        # h = 10
        # fig = plt.figure(figsize=(8, 8))
        # columns = 2
        # rows = 1
        # fig.add_subplot(rows, columns, 1)
        # plt.imshow(numpy_image)

        # fig.add_subplot(rows, columns, 2)
        # plt.imshow(mask)
        # plt.show()

        # Load label
        match = re.search(self.pattern_rgb, image_file).group(1)
        label = np.load(label_file)["arr_0"].astype(np.uint8)

        if idx % 2 == 0:
            if label == 0:
                label = np.array([1]).astype(np.uint8)
            elif label == 1:
                label = np.array([0]).astype(np.uint8)
            elif label == 2:
                label = np.array([2]).astype(np.uint8)
            elif label == 3:
                label = np.array([4]).astype(np.uint8)
            elif label == 4:
                label = np.array([3]).astype(np.uint8)

        image = torch.from_numpy(numpy_image.copy()).permute(2, 0, 1).float().div(255)
        image = self.normalize_transform(image)
        image = self.resize_transform(image)

        mask = torch.from_numpy(mask.copy()).unsqueeze(0)
        mask = self.resize_transform(mask)

        return image, label, mask

    def __len__(self):
        return len(self.image_label_pairs)


def init_model():
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")
    model = model.to("cuda")

    for name, param in model.named_parameters():
        if "pretrained" in name or "depth_head.scratch.output_conv" in name:
            param.requires_grad = False
        # if "nbv" in name or "affordance" in name:
        #     continue
        # param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-1
    # )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    return model, criterion, optimizer, scheduler


def train(train_dataloader, val_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epoch = 100

    model, criterion, optimizer, scheduler = init_model()
    # progress_bar = tqdm(range(num_epoch*len(train_dataloader) + num_epoch*len(val_dataloader)))

    running_loss = 0.0

    scaling_factor = 1

    model.train()

    lowest_loss = 10000.0

    for epoch in range(num_epoch):
        running_loss = 0.0

        for images, label_x_y, mask in train_dataloader:
            images, label_x_y, mask = (
                images.to(device),
                label_x_y.to(device),
                mask.to(device),
            )
            optimizer.zero_grad()

            nbv_out = model(images, mask)
            label_x_y = label_x_y.squeeze()
            # labels = labels.squeeze().type(torch.float).cuda()
            loss = criterion(nbv_out, label_x_y).requires_grad_(True)
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
            for images, label_x_y, mask in val_dataloader:
                images, label_x_y, mask = (
                    images.to(device),
                    label_x_y.to(device),
                    mask.to(device),
                )

                nbv_out = model(images, mask)
                label_x_y = label_x_y.squeeze()
                loss = criterion(nbv_out, label_x_y)
                scaled_loss = loss * scaling_factor

                validation_loss += scaled_loss.item()
                # progress_bar.update(1)
        nbv_out = torch.argmax(nbv_out, dim=1)
        print(nbv_out, label_x_y)
        avg_validation_loss = validation_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_validation_loss}")
        if avg_validation_loss < lowest_loss:
            lowest_loss = avg_validation_loss
            torch.save(
                model,
                "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/Depth-Anything/huggingface/Depth-Anything/checkpoint_finetune_nbv_9_views.pth",
            )

        model.train()

        scheduler.step(avg_validation_loss)


if __name__ == "__main__":
    # Set the start method to 'spawn'
    mp.set_start_method("spawn", force=True)

    dataset = CustomDataset(
        labels_root="/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/labels_nbv_3F_2_9_views",
        images_root="/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/labels_nbv_3F_2_9_views",
    )

    print(len(dataset))

    # Determine sizes for training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=8
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=8
    )

    # Example training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(train_dataloader, val_dataloader)
