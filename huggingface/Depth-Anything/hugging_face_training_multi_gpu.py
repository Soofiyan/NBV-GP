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
                for image_file in glob.glob(os.path.join(folder_id_path, "rgb*.jpg")):
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
                    segmask_file = image_file.replace("rgb", "segmask").replace(".jpg", ".png")
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
        numpy_image = cv2.imread(image_file).astype(np.uint8)
        crop_bin_dim = [73, 287, 195, 442]

        numpy_image = CropTransform(crop_bin_dim)(numpy_image)
        numpy_image = np.array(numpy_image).astype(np.float32) / 255.0
        image = (
            torch.from_numpy(numpy_image).permute(2, 0, 1).float().div(255)
        )
        image = self.normalize_transform(image)
        image = self.resize_transform(image)

        # Load label
        label = Image.open(label_file).convert("L")
        numpy_label = np.array(label)
        label = torch.from_numpy(numpy_label).unsqueeze(0).float().div(255)
        label = self.resize_transform(label)

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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def weighted_huber_loss(predictions, labels, delta=1.0, epsilon=1e-3):
    # Calculate the errors
    errors = predictions - labels
    # Define weights; here we use 1/(scores + epsilon) to avoid division by zero
    weights = 1 / (torch.abs(errors) + epsilon)
    # Calculate Huber loss
    is_small_error = torch.abs(errors) < delta
    squared_loss = 0.5 * (errors ** 2)
    linear_loss = delta * (torch.abs(errors) - 0.5 * delta)
    # Combine using a mask
    huber_loss = torch.where(is_small_error, squared_loss, linear_loss)
    weighted_huber = weights * huber_loss
    # Mean over all samples
    loss = torch.mean(weighted_huber)
    return loss



def init_model():
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")
    # model.load_state_dict(
    #     torch.load(
    #         "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/Depth-Anything/huggingface/Depth-Anything/temp.pth"
    #     )
    # )

    for name, param in model.named_parameters():
        if "pretrained" in name or "depth_head.scratch.output_conv" in name:
            param.requires_grad = False
        # print(name, param.requires_grad)

    criterion = torch.nn.HuberLoss()
    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-1
    # )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    return model, criterion, optimizer, scheduler


def train(rank, world_size, train_dataloader, val_dataloader):

    setup(rank, world_size)

    dataset = CustomDataset(
        images_root="/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/temp",
        labels_root="/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/labels_nbv_3F",
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # Determine sizes for training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0
    )

    num_epoch = 50

    model, criterion, optimizer, scheduler = init_model()

    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    # progress_bar = tqdm(range(num_epoch*len(train_dataloader) + num_epoch*len(val_dataloader)))

    running_loss = 0.0

    scaling_factor = 10e3

    model.train()

    lowest_loss = float("inf")

    for epoch in range(num_epoch):
        running_loss = 0.0

        for images, labels, mask in train_dataloader:
            images, labels, mask = images.to(rank), labels.to(rank), mask.to(rank)
            optimizer.zero_grad()
            
            depth_out, affordance_map_out = model(images, mask)
            
            weighted_loss = weighted_huber_loss(affordance_map_out, labels)
            scaled_loss = weighted_loss * scaling_factor

            scaled_loss.backward()
            optimizer.step()
            running_loss += scaled_loss.item()

            # progress_bar.update(1)
            # print(f"Loss: {loss.item()}")

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

        model.eval()  # Set the model to evaluation mode
        validation_loss = 0.0

        with torch.no_grad():  # Disable gradient computation
            for images, labels, mask in val_dataloader:
                images, labels, mask = (
                    images.to(rank),
                    labels.to(rank),
                    mask.to(rank),
                )
                
                _, affordance_map_out = model(images, mask)
                # loss = criterion(affordance_map_out, labels)
                weighted_loss = weighted_huber_loss(affordance_map_out, labels)
                scaled_loss = weighted_loss * scaling_factor

                validation_loss += scaled_loss.item()

                # progress_bar.update(1)
        avg_validation_loss = validation_loss / len(val_dataloader)
        if avg_validation_loss < lowest_loss:
            lowest_loss = avg_validation_loss
            torch.save(
            	model,
               "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/Depth-Anything/huggingface/Depth-Anything/temp.pth",
            )
        print(f"Validation Loss: {avg_validation_loss}")

        save_plot(labels[0], affordance_map_out[0], mask[0], epoch)

        model.train()

        scheduler.step(avg_validation_loss)
    
    cleanup()


if __name__ == "__main__":
    # Set the start method to 'spawn'
    # mp.set_start_method("spawn", force=True)

    world_size = torch.cuda.device_count()


    images_root="/home/ubuntu/temp"
    labels_root="/home/ubuntu/labels_nbv_3F"

    

    # Example training loop
    mp.spawn(train, args=(world_size, images_root, labels_root), nprocs=world_size, join=True)

    # train(rank, world_size, train_dataloader, val_dataloader)
