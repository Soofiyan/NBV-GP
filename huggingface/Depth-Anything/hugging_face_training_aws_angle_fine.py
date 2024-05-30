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

    def create_rhombus_mask(self, image, vertices):
        # Create a mask with zeros (black)
        mask = np.zeros_like(image)
        # Define the rhombus vertices - these need to be specified as a list of points (x, y)
        points = np.array([vertices], dtype=np.int32)
        # Fill the rhombus with white (255)
        cv2.fillPoly(mask, points, (255, 255, 255))
        return mask

    def crop_image(self, image, mask, pad_size=50):
        # Apply mask to the image (single-channel)

        mask = (mask / 255).astype(np.uint8)
        masked_image = image * mask
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = masked_image[y : y + h, x : x + w]
        # Padding color is 0 for single-channel images
        # padded_image = cv2.copyMakeBorder(cropped_image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)
        return cropped_image

    def __call__(self, x):
        mask_rgb = self.create_rhombus_mask(x, self.crop_dim)
        rgb_image_cropped = self.crop_image(x, mask_rgb, pad_size=50)
        return rgb_image_cropped


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

        self.image_label_pairs = self._get_image_label_pairs()

        self.pattern = r"affordance_map_(\d+)_"
        self.crop_bin_dim_dict = {
            "0": [(239, 102), (465, 112), (447, 295), (237, 307)],
            "1": [(225, 100), (463, 107), (447, 299), (227, 309)],
            "2": [(193, 102), (447, 102), (437, 305), (206, 308)],
            "3": [(179, 109), (415, 100), (415, 309), (194, 299)],
            "4": [(176, 113), (401, 102), (406, 309), (194, 296)],
            "5": [(239, 69), (465, 85), (463, 276), (238, 291)],
            "6": [(224, 71), (460, 81), (460, 281), (223, 294)],
            "7": [(196, 74), (443, 72), (445, 291), (197, 294)],
            "8": [(180, 82), (416, 71), (416, 294), (182, 285)],
            "9": [(178, 90), (402, 73), (405, 294), (180, 278)],
            "10": [(236, 55), (447, 68), (465, 250), (238, 262)],
            "11": [(225, 54), (447, 64), (464, 256), (224, 260)],
            "12": [(203, 55), (436, 54), (447, 259), (194, 259)],
            "13": [(192, 64), (414, 55), (414, 262), (175, 255)],
            "14": [(192, 71), (402, 54), (402, 263), (175, 253)],
        }

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
                    segmask_file = image_file.replace("rgb", "segmask").replace(
                        ".jpg", ".png"
                    )
                    for label_file in glob.glob(label_pattern):
                        if (
                            "affordance_map_roll" in label_file
                            or "affordance_map_pitch" in label_file
                        ):
                            continue
                        label_img = Image.open(label_file).convert("L")
                        label_roll_file = label_file.replace(
                            "affordance_map", "affordance_map_roll"
                        )
                        label_pitch_file = label_file.replace(
                            "affordance_map", "affordance_map_pitch"
                        )
                        if np.any(np.array(label_img) > 0):
                            pairs.append(
                                (
                                    image_file,
                                    label_file,
                                    label_roll_file,
                                    label_pitch_file,
                                    segmask_file,
                                )
                            )
                        # pairs.append((image_file, label_file, segmask_file))
        return pairs

    def __getitem__(self, idx):
        image_file, label_file, label_roll_file, label_pitch_file, segmask_file = (
            self.image_label_pairs[idx]
        )

        # Load image
        numpy_image = cv2.imread(image_file).astype(np.uint8)
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

        crop_bin_dim = [73, 287, 195, 442]
        numpy_image = numpy_image[
            crop_bin_dim[0] : crop_bin_dim[1], crop_bin_dim[2] : crop_bin_dim[3]
        ]
        image = torch.from_numpy(numpy_image).permute(2, 0, 1).float().div(255)
        image = self.normalize_transform(image)
        image = self.resize_transform(image)

        numpy_mask = cv2.imread(segmask_file, 0)
        match = re.search(self.pattern, label_file)
        mask_id = match.group(1)
        numpy_mask = numpy_mask[
            crop_bin_dim[0] : crop_bin_dim[1], crop_bin_dim[2] : crop_bin_dim[3]
        ]
        numpy_mask = numpy_mask == int(mask_id)
        mask = torch.from_numpy(numpy_mask).unsqueeze(0)
        mask = self.resize_transform(mask)

        # Load label
        label = Image.open(label_file).convert("L")
        numpy_label = np.array(label)
        label = torch.from_numpy(numpy_label).unsqueeze(0).float().div(255)
        label = self.resize_transform(label)

        label_roll = Image.open(label_roll_file).convert("L")
        numpy_label_roll = np.array(label_roll)
        label_roll = torch.from_numpy(numpy_label_roll).unsqueeze(0).float().div(255)
        label_roll = self.resize_transform(label_roll)

        label_pitch = Image.open(label_pitch_file).convert("L")
        numpy_label_pitch = np.array(label_pitch)
        label_pitch = torch.from_numpy(numpy_label_pitch).unsqueeze(0).float().div(255)
        label_pitch = self.resize_transform(label_pitch)

        return image, label, label_roll, label_pitch, mask

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


def weighted_huber_loss(predictions, labels, delta=1.0, epsilon=2e-3):
    # Calculate the errors
    errors = predictions - labels
    # Define weights; here we use 1/(scores + epsilon) to avoid division by zero
    weights = 1 / (torch.abs(errors) + epsilon)
    # Calculate Huber loss
    is_small_error = torch.abs(errors) < delta
    squared_loss = 0.5 * (errors**2)
    linear_loss = delta * (torch.abs(errors) - 0.5 * delta)
    # Combine using a mask
    huber_loss = torch.where(is_small_error, squared_loss, linear_loss)
    weighted_huber = weights * huber_loss
    # Mean over all samples
    loss = torch.mean(weighted_huber)
    return loss


def init_model():
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")

    for name, param in model.named_parameters():
        if "pretrained" in name or "depth_head.scratch.output_conv" in name:
            param.requires_grad = False
        # print(name, param.requires_grad)

    criterion = torch.nn.HuberLoss()
    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-1
    # )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=7)

    return model, criterion, optimizer, scheduler


def train(rank, world_size, images_root_path, labels_root_path):

    setup(rank, world_size)

    dataset = CustomDataset(
        images_root=images_root_path,
        labels_root=labels_root_path,
    )
    print(len(dataset))

    # Determine sizes for training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    sampler_train = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    sampler_val = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    local_batch_size = 32

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=sampler_train,
        num_workers=32,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=local_batch_size,
        sampler=sampler_val,
        num_workers=32,
        pin_memory=True,
    )

    num_epoch = 200

    model, criterion, optimizer, scheduler = init_model()

    # model = torch.load(
    #     "/home/ubuntu/checkpoint_finetune2.pth",
    #     map_location=lambda storage, loc: storage.cuda(rank),
    # )
    # model = model.module

    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    running_loss = 0.0

    scaling_factor = 10e3

    model.train()

    lowest_loss = 100000.0

    for epoch in range(num_epoch):
        running_loss = 0.0
        sampler_train.set_epoch(epoch)
        sampler_val.set_epoch(epoch)

        for (
            images,
            labels,
            labels_angle_roll,
            labels_angle_pitch,
            mask,
        ) in train_dataloader:
            images, labels, labels_angle_roll, labels_angle_pitch, mask = (
                images.to(rank),
                labels.to(rank),
                labels_angle_roll.to(rank),
                labels_angle_pitch.to(rank),
                mask.to(rank),
            )
            optimizer.zero_grad()

            (
                depth_out,
                affordance_map_out,
                affordance_map_roll_out,
                affordance_map_pitch_out,
            ) = model(images, mask)

            loss = criterion(affordance_map_out, labels)
            loss_angle_roll = criterion(affordance_map_roll_out, labels_angle_roll)
            loss_angle_pitch = criterion(affordance_map_pitch_out, labels_angle_pitch)
            scaled_loss = (
                loss * scaling_factor
                + loss_angle_roll * scaling_factor
                + loss_angle_pitch * scaling_factor
            )

            scaled_loss.backward()
            optimizer.step()
            running_loss += scaled_loss.item()

            # progress_bar.update(1)
            # print(f"Loss: {loss.item()}")

        epoch_loss = running_loss / len(train_dataloader)
        if rank == 0:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

        model.eval()  # Set the model to evaluation mode
        validation_loss = 0.0

        with torch.no_grad():  # Disable gradient computation
            for (
                images,
                labels,
                labels_angle_roll,
                labels_angle_pitch,
                mask,
            ) in val_dataloader:
                images, labels, labels_angle_roll, labels_angle_pitch, mask = (
                    images.to(rank),
                    labels.to(rank),
                    labels_angle_roll.to(rank),
                    labels_angle_pitch.to(rank),
                    mask.to(rank),
                )
                (
                    depth_out,
                    affordance_map_out,
                    affordance_map_roll_out,
                    affordance_map_pitch_out,
                ) = model(images, mask)

                loss = criterion(affordance_map_out, labels)
                loss_angle_roll = criterion(affordance_map_roll_out, labels_angle_roll)
                loss_angle_pitch = criterion(
                    affordance_map_pitch_out, labels_angle_pitch
                )
                scaled_loss = (
                    loss * scaling_factor
                    + loss_angle_roll * scaling_factor
                    + loss_angle_pitch * scaling_factor
                )

                validation_loss += scaled_loss.item()

                # progress_bar.update(1)
        avg_validation_loss = validation_loss / len(val_dataloader)
        if rank == 0 and avg_validation_loss < lowest_loss:
            lowest_loss = avg_validation_loss
            torch.save(
                model,
                "/home/ubuntu/checkpoint_multi_angle_30_c.pth",
            )
        if rank == 0:
            print(f"Validation Loss: {avg_validation_loss}")

        # save_plot(labels[0], affordance_map_out[0], mask[0], epoch)

        model.train()

        scheduler.step(avg_validation_loss)

    cleanup()


if __name__ == "__main__":
    # Set the start method to 'spawn'
    mp.set_start_method("fork", force=True)

    world_size = torch.cuda.device_count()

    images_root = "/home/ubuntu/compress_observation"
    labels_root = "/home/ubuntu/labels_nbv_3F_angle_30_c"

    # Example training loop
    mp.spawn(
        train, args=(world_size, images_root, labels_root), nprocs=world_size, join=True
    )
