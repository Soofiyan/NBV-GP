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
        ).to("cuda")
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ).to("cuda")

        self.image_label_pairs = self._get_image_label_pairs()

        self.pattern = r"affordance_map_(\d+)_"

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
                for image_file in glob.glob(os.path.join(folder_id_path, "rgb*.npy")):
                    # Derive the base name for the image file to match labels
                    base_name = os.path.splitext(os.path.basename(image_file))[0]
                    count = base_name.split("_")[-1]
                    # Pattern for corresponding label files
                    label_pattern = os.path.join(
                        self.labels_root,
                        run_id,
                        "3F",
                        folder_id,
                        f"affordance_map_*_{folder_id}_{count}.png",
                    )
                    # Find all matching label files and add to pairs
                    segmask_file = image_file.replace("rgb", "segmask")
                    for label_file in glob.glob(label_pattern):
                        label_img = Image.open(label_file).convert("L")
                        if (
                            np.any(np.array(label_img) > 0)
                        ):
                            pairs.append((image_file, label_file, segmask_file))
                        # pairs.append((image_file, label_file, segmask_file))
        return pairs

    def __getitem__(self, idx):
        image_file, label_file, segmask_file = self.image_label_pairs[idx]

        # Load image
        numpy_image = np.load(image_file).astype(np.uint8)
        crop_bin_dim = [73, 315, 195, 442]

        numpy_image = CropTransform(crop_bin_dim)(numpy_image)
        raw_image = numpy_image
        numpy_image = np.array(numpy_image).astype(np.float32) / 255.0
        image = (
            torch.from_numpy(numpy_image).permute(2, 0, 1).float().div(255).to("cuda")
        )
        image = self.normalize_transform(image)
        image = self.resize_transform(image)

        # Load label
        label = Image.open(label_file).convert("L")
        numpy_label = np.array(label)
        label = torch.from_numpy(numpy_label).unsqueeze(0).float().div(255).to("cuda")
        label = self.resize_transform(label)

        numpy_mask = np.load(segmask_file)
        match = re.search(self.pattern, label_file)
        mask_id = match.group(1)
        numpy_mask = CropTransform(crop_bin_dim)(numpy_mask)
        mask = numpy_mask == int(mask_id)
        mask = torch.from_numpy(mask).unsqueeze(0).to("cuda")
        mask = self.resize_transform(mask)

        return raw_image, image, label, mask

    def __len__(self):
        return len(self.image_label_pairs)
        

def save_plot(label, affordance_map, mask, epoch):
    fig, axs = plt.subplots(1, 3)
    # Plot the first subplot
    axs[0].imshow(mask.squeeze(0).detach().cpu().numpy())  # red line
    axs[0].axis("off")

    axs[1].imshow(label.squeeze(0).detach().cpu().numpy())  # red line
    axs[1].axis("off")

    axs[2].imshow(affordance_map.squeeze(0).detach().cpu().numpy())  # red line
    axs[2].axis("off")

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    save_path = "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/results_plots"
    plt.savefig(f"{save_path}/epoch_{epoch}.png")

def view_plot(label, affordance_map, mask):
    fig, axs = plt.subplots(1, 3)
    # Plot the first subplot
    axs[0].imshow(mask.squeeze(0).detach().cpu().numpy())  # red line
    axs[0].axis("off")

    axs[1].imshow(label.squeeze(0).detach().cpu().numpy())  # red line
    axs[1].axis("off")

    axs[2].imshow(affordance_map.squeeze(0).detach().cpu().numpy())  # red line
    axs[2].axis("off")

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()


def init_model():
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")
    model.load_state_dict(
        torch.load(
            "checkpoint_finetune.pth"
        )
    )
    model = model.to("cuda")

    for name, param in model.named_parameters():
        if "pretrained" in name or "depth_head.scratch.output_conv" in name:
            param.requires_grad = False
        # print(name, param.requires_grad)

    criterion = torch.nn.HuberLoss()

    return model, criterion


def test(val_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epoch = 50

    model, criterion = init_model()

    scaling_factor = 10e3

    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _ in range(10):
        with torch.no_grad():  # Disable gradient computation
            for raw_images, images, labels, mask in val_dataloader:
                images, labels, mask = (
                    images.to(device),
                    labels.to(device),
                    mask.to(device),
                )
                _, affordance_map_out = model(images, mask)
                affordance_max_value = torch.max(affordance_map_out.squeeze(0).squeeze(0))
                affordance_coordiantes = torch.where(affordance_map_out.squeeze(0).squeeze(0) == affordance_max_value)

                print(affordance_coordiantes)
                labels_max_value = torch.max(labels.squeeze(0).squeeze(0))
                labels_coordiantes = torch.where(labels.squeeze(0).squeeze(0) == labels_max_value)
                raw_images = raw_images.cpu().detach().numpy()
                resize_factor_x = 242 / 518
                resize_factor_y = 247 / 518
                cv2.circle(raw_images[0], (int(affordance_coordiantes[1].item()*resize_factor_x), int(affordance_coordiantes[0].item()*resize_factor_y)), 2, (0, 0, 255), -1)
                cv2.circle(raw_images[0], (int(labels_coordiantes[1].item()*resize_factor_x), int(labels_coordiantes[0].item()*resize_factor_x)), 2, (0, 255, 0), -1)
                cv2.imshow("image", raw_images[0])
                cv2.waitKey(0)
                view_plot(labels[0], affordance_map_out[0], mask[0])


                loss = criterion(affordance_map_out, labels).to("cuda")
                scaled_loss = loss * scaling_factor

    print(f"Validation Loss: {scaled_loss}")


if __name__ == "__main__":
    # Set the start method to 'spawn'
    mp.set_start_method("spawn", force=True)

    dataset = CustomDataset(
        images_root="/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/test",
        labels_root="/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/labels_nbv_3F",
    )

    # Determine sizes for training and validation sets
    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    # Example training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test(val_dataloader)
