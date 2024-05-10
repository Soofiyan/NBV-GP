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


import torch.distributed as dist

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
        image_file = "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/compress_observation/run_ID_14/3F/7/rgb_7_10.jpg"
        segmask_file = "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/compress_observation/run_ID_14/3F/7/segmask_7_10.png"
        pairs.append((image_file, segmask_file))
        return pairs

    def __getitem__(self, idx):
        image_file, segmask_file = self.image_label_pairs[idx]

        # Load image
        numpy_image = cv2.imread(image_file).astype(np.uint8)
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        
        crop_bin_dim = [74, 291, 196, 445]

        numpy_image = CropTransform(crop_bin_dim)(numpy_image)
        raw_image = numpy_image
        numpy_image = np.array(numpy_image).astype(np.float32)
        image = (
            torch.from_numpy(numpy_image).permute(2, 0, 1).float().div(255).to("cuda")
        )
        image = self.resize_transform(image)
        image = self.normalize_transform(image)

        numpy_mask = cv2.imread(segmask_file, 0)
        plt.imshow(numpy_mask)
        plt.show()
        for mask_id in np.unique(numpy_mask):
            if(mask_id != 9):
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
    save_path = "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/results_plots"
    plt.savefig(f"{save_path}/epoch_{epoch}.png")

def view_plot(affordance_map, mask):
    fig, axs = plt.subplots(1, 2)
    # Plot the first subplot
    axs[0].imshow(mask.squeeze(0).detach().cpu().numpy())  # red line
    axs[0].axis("off")

    # axs[1].imshow(label.squeeze(0).detach().cpu().numpy())  # red line
    # axs[1].axis("off")

    axs[1].imshow(affordance_map.squeeze(0).detach().cpu().numpy())  # red line
    axs[1].axis("off")

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()


def init_model():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")
    dist.init_process_group(backend='nccl', init_method='env://', rank=0, world_size=1)
    # model = torch.load(
    #         "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/Depth-Anything/huggingface/Depth-Anything/weights/checkpoint_finetune2.pth"
    #     )
    model = torch.load("/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/Depth-Anything/huggingface/Depth-Anything/weights/checkpoint_finetune_multi_view.pth")
    # new_state_dict = {k[len("module."):]: v for k, v in model_state_dict.items()}
    # model.load_state_dict(new_state_dict)
    model = model.to("cuda")
    model = model.module


    # model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vits14")
    # model_state_dict = torch.load("/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/Depth-Anything/huggingface/Depth-Anything/weights/initial_training.pth")
    # model.load_state_dict(model_state_dict)
    # model = model.to("cuda")

    for name, param in model.named_parameters():
        if "pretrained" in name or "depth_head.scratch.output_conv" in name:
            param.requires_grad = False
        # print(name, param.requires_grad)

    criterion = torch.nn.HuberLoss()

    return model, criterion


def test(val_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, criterion = init_model()

    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _ in range(10):
        with torch.no_grad():  # Disable gradient computation
            for raw_images, images, mask in val_dataloader:
                images, mask = (
                    images.to(device),
                    mask.to(device),
                )
                _, affordance_map_out = model(images, mask)

                affordance_max_value = torch.max(affordance_map_out.squeeze(0).squeeze(0))
                affordance_coordiantes = torch.where(affordance_map_out.squeeze(0).squeeze(0) == affordance_max_value)
                
                print(affordance_coordiantes)
                # labels_max_value = torch.max(labels.squeeze(0).squeeze(0))
                # labels_coordiantes = torch.where(labels.squeeze(0).squeeze(0) == labels_max_value)
                raw_images = raw_images.cpu().detach().numpy()[0]
                raw_images = cv2.resize(raw_images, (224, 224))
                shape_raw = raw_images.shape
                resize_factor_x = shape_raw[0] / 224
                resize_factor_y = shape_raw[1] / 224

                cv2.circle(raw_images, (int(affordance_coordiantes[1].item()*resize_factor_x), int(affordance_coordiantes[0].item()*resize_factor_y)), 2, (0, 0, 255), -1)
                # cv2.circle(raw_images, (int(labels_coordiantes[1].item()*resize_factor_x), int(labels_coordiantes[0].item()*resize_factor_x)), 2, (0, 255, 0), -1)
                cv2.imshow("image", raw_images)
                cv2.waitKey(0)
                view_plot(affordance_map_out[0], mask[0])
    dist.destroy_process_group()


if __name__ == "__main__":
    # Set the start method to 'spawn'
    mp.set_start_method("spawn", force=True)

    dataset = CustomSingleDataset()

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [0, 1])

    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    # Example training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test(val_dataloader)
