import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

from depth_anything.dpt_angle_fine import DepthAnything
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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


class CustomSingleDataset(Dataset):
    def __init__(self):

        self.resize_transform = transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
        ).to("cuda")
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ).to("cuda")

        self.image_label_pairs = self._get_image_label_pairs()

    def _get_image_label_pairs(self):
        pairs = []
        image_file = "/home/soofiyan_ws/Documents/VLM_Grasp/VLM_Grasp/data/mini_val/rgb_images/002_rgb_image.png"
        segmask_file = "/home/soofiyan_ws/Documents/VLM_Grasp/VLM_Grasp/data/mini_val/binary_masks/002_1_mask.png"
        pairs.append((image_file, segmask_file))
        return pairs

    def __getitem__(self, idx):
        image_file, segmask_file = self.image_label_pairs[idx]

        numpy_image = cv2.imread(image_file).astype(np.uint8)
        plt.imshow(numpy_image)
        plt.show()
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        # crop_bin_dim = [74, 291, 196, 445]
        crop_bin_dim = [0, 221, 0, 332]
        numpy_image = CropTransform(crop_bin_dim)(numpy_image)
        image = torch.from_numpy(numpy_image).permute(2, 0, 1).float().div(255)
        image = self.normalize_transform(image)
        image = self.resize_transform(image)

        raw_image = (
            self.resize_transform(torch.from_numpy(numpy_image).permute(2, 0, 1))
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        numpy_mask = cv2.imread(segmask_file, 0)
        plt.imshow(numpy_mask)
        plt.show()
        for mask_id in np.unique(numpy_mask):
            if mask_id != 255:
                continue
            numpy_mask_tensor = torch.from_numpy(numpy_mask).to("cuda")
            numpy_mask_tensor = CropTransform(crop_bin_dim)(numpy_mask_tensor)
            mask = numpy_mask_tensor == int(mask_id)
            mask = mask.unsqueeze(0).float()
            mask = self.resize_transform(mask)
            # if(random.random() < 0.5):
            #     break

        return raw_image, image, mask

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
    save_path = "/home/ubuntu/results_plots_2"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/epoch_{epoch}.png")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def init_model():
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")
    for name, param in model.named_parameters():
        if "pretrained" in name or "depth_head.scratch.output_conv" in name:
            param.requires_grad = False
    criterion = torch.nn.HuberLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=7)
    return model, criterion, optimizer, scheduler


def train(rank, world_size):

    setup(rank, world_size)

    dataset = CustomSingleDataset()

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [0, 1])

    inference_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    model, criterion, optimizer, scheduler = init_model()

    model = torch.load(
        "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/Depth-Anything/huggingface/Depth-Anything/weights/checkpoint_multi_angle_30.pth",
    )
    model = model.module
    model.to("cuda")

    model.eval()

    with torch.no_grad():  # Disable gradient computation
        for (
            raw_image,
            images,
            mask,
        ) in inference_dataloader:
            raw_image, images, mask = (
                raw_image,
                images.to(rank),
                mask.to(rank),
            )
            (
                depth_out,
                affordance_map_out,
                affordance_map_roll_out,
                affordance_map_pitch_out,
            ) = model(images, mask)

            affordance_map_out = (
                affordance_map_out.squeeze(0).squeeze(0).detach().cpu().numpy()
            )

            from sklearn.cluster import DBSCAN

            # Flatten the affordance map for clustering
            X = np.column_stack(np.nonzero(affordance_map_out))
            scores = affordance_map_out[X[:, 0], X[:, 1]]

            # DBSCAN clustering
            db = DBSCAN(eps=1.5, min_samples=5).fit(X, sample_weight=scores)
            labels = db.labels_

            # Analyze clusters
            unique_labels = set(labels)
            clusters = {}
            for label in unique_labels:
                if label == -1:
                    continue  # Ignore noise
                cluster_points = X[labels == label]
                cluster_scores = scores[labels == label]
                average_score = np.mean(cluster_scores)
                clusters[label] = {
                    "points": cluster_points,
                    "scores": cluster_scores,
                    "average_score": average_score,
                    "size": len(cluster_points),
                }

            # Find the best cluster
            best_cluster_label = max(clusters, key=lambda label: clusters[label]['average_score'])
            best_cluster = clusters[best_cluster_label]

            # Sort points in the best cluster by score and limit to max_points
            sorted_indices = np.argsort(best_cluster['scores'])[::-1]  # Sort in descending order
            limited_indices = sorted_indices[:175]
            limited_points = best_cluster['points'][limited_indices]
            limited_scores = best_cluster['scores'][limited_indices]

            # Calculate the centroid of the limited points
            centroid = np.mean(limited_points, axis=0)
            centroid_x = int(centroid[0])
            centroid_y = int(centroid[1])

            # Display the centroid
            print(f"Centroid coordinate: ({centroid_x}, {centroid_y})")

            fig, axs = plt.subplots(1, 5, figsize=(12, 3))

            affordance_map_pitch_out = (
                affordance_map_pitch_out.squeeze(0).squeeze(0).detach().cpu().numpy()
            )
            affordance_map_roll_out = (
                affordance_map_roll_out.squeeze(0).squeeze(0).detach().cpu().numpy()
            )

            max_idx = np.unravel_index(
                affordance_map_out.argmax(), affordance_map_out.shape
            )
            max_y, max_x = max_idx  # Coordinates of the maximum value
            print(max_y, max_x)
            print(
                affordance_map_out[max_y, max_x],
                affordance_map_roll_out[max_y, max_x] * 180 / np.pi - 30,
                affordance_map_pitch_out[max_y, max_x] * 180 / np.pi - 30,
            )
            axs[0].imshow(affordance_map_out)
            axs[0].add_patch(
                plt.Circle((max_x, max_y), radius=2, color="red", fill=True)
            )
            axs[0].axis("off")

            axs[1].imshow(affordance_map_roll_out)
            axs[1].axis("off")

            axs[2].imshow(affordance_map_pitch_out)
            axs[2].axis("off")

            axs[3].imshow(raw_image.squeeze(0))
            axs[3].add_patch(
                plt.Circle((max_x, max_y), radius=2, color="red", fill=True)
            )
            axs[3].add_patch(
                plt.Circle((centroid_y, centroid_x), radius=2, color="blue", fill=True)
            )
            axs[3].axis("off")

            axs[4].imshow(mask.squeeze(0).squeeze(0).detach().cpu().numpy())
            axs[4].add_patch(
                plt.Circle((max_x, max_y), radius=2, color="black", fill=True)
            )
            axs[4].axis("off")

            plt.tight_layout()  # Adjust subplots to fit into the figure area.
            plt.show()

    cleanup()


if __name__ == "__main__":
    # Set the start method to 'spawn'
    mp.set_start_method("fork", force=True)

    world_size = torch.cuda.device_count()

    images_root = "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/temp"
    labels_root = "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/aws_processed/labels_nbv_3F_angle_30"

    train(0, world_size)
