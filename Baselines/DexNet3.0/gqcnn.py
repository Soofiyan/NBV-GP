import torch
import torch.nn as nn
import torch.nn.functional as F
import os


from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import numpy as np
from PIL import Image
import cv2
import re
import random
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


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

        self.batch_size = 64
        self.patch_size = 32
        self.pattern = r"affordance_map_(\d+)_"

        self.image_label_pairs = self._get_image_label_pairs()

    def _get_image_label_pairs(self):
        pairs = []
        # Iterate through all the run_ID directories
        for run_id_path in glob.glob(os.path.join(self.images_root, "run_ID_*")):
            # Extract run ID from the path
            run_id = os.path.basename(run_id_path)
            run_id_number = run_id.split("_")[-1]
            if int(run_id_number) > 50:
                continue
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

                    depth_file = os.path.join(
                        "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/observation_data_nbv_3F",
                        run_id,
                        "3F",
                        folder_id,
                        f"depth_image_{folder_id}_{count}.npy",
                    )
                    # Find all matching label files and add to pairs
                    segmask_file = image_file.replace("rgb", "segmask").replace(
                        ".jpg", ".png"
                    )
                    for label_file in glob.glob(label_pattern):
                        label_img = Image.open(label_file).convert("L")
                        if np.any(np.array(label_img) > 0):
                            pairs.append(
                                (image_file, label_file, segmask_file, depth_file)
                            )
                        # pairs.append((image_file, label_file, segmask_file))
        return pairs

    def _get_patch_indices(self, segmask_file, label_file):
        numpy_mask = cv2.imread(segmask_file, cv2.IMREAD_GRAYSCALE)
        match = re.search(self.pattern, label_file)
        mask_id = match.group(1)
        crop_bin_dim = [73, 287, 195, 442]
        padding_size = [
            self.patch_size,
            self.patch_size,
            self.patch_size,
            self.patch_size,
        ]
        numpy_mask = CropTransform(crop_bin_dim)(numpy_mask)
        mask = numpy_mask == int(mask_id)
        mask = torch.from_numpy(mask).unsqueeze(0)
        mask = self.resize_transform(mask)
        mask = F.pad(mask, padding_size, "constant", 0).squeeze(0)

        non_zero_coords = torch.nonzero(mask, as_tuple=False)
        non_zero_coords = [tuple(coord.tolist()) for coord in non_zero_coords]
        non_zero_coords = random.sample(
            non_zero_coords, min(len(non_zero_coords), self.batch_size)
        )

        return non_zero_coords

    def _extract_patch(self, image, label, path_idx, pad_size):
        y, x = path_idx
        patch = image[
            y - self.patch_size // 2 : y + self.patch_size // 2,
            x - self.patch_size // 2 : x + self.patch_size // 2,
        ]
        label = label[y, x]
        return patch, label

    def __getitem__(self, idx):
        image_file, label_file, segmask_file, depth_file = self.image_label_pairs[idx]
        patch_indices = self._get_patch_indices(segmask_file, label_file)

        batch_patches = []
        batch_labels = []
        crop_bin_dim = [73, 287, 195, 442]
        padding_size = [
            self.patch_size,
            self.patch_size,
            self.patch_size,
            self.patch_size,
        ]

        for patch_idx in patch_indices:
            # Load and preprocess image
            # numpy_image = cv2.imread(image_file).astype(np.uint8)
            # numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
            # numpy_image = CropTransform(crop_bin_dim)(numpy_image)
            # image = torch.from_numpy(numpy_image).permute(2, 0, 1).float().div(255)
            # # image = self.normalize_transform(image)
            # image = self.resize_transform(image)
            # image = F.pad(image, 32, "constant", 0).squeeze(0)

            depth_image = np.load(depth_file)
            depth_image = CropTransform(crop_bin_dim)(depth_image)
            depth_image = torch.from_numpy(depth_image).unsqueeze(0).float()
            depth_image = self.resize_transform(depth_image)
            depth_image = F.pad(depth_image, padding_size, "constant", 0).squeeze(0)

            # Load label
            label = Image.open(label_file).convert("L")
            numpy_label = np.array(label)
            label = torch.from_numpy(numpy_label).unsqueeze(0).float().div(255)
            label = self.resize_transform(label)
            label = F.pad(label, padding_size, "constant", 0).squeeze(0)

            patch, patch_label = self._extract_patch(
                depth_image, label, patch_idx, padding_size[0]
            )
            batch_patches.append(patch)
            batch_labels.append(patch_label)

        batch_patches = torch.stack(batch_patches)
        batch_labels = torch.stack(batch_labels)

        return batch_patches, batch_labels

    def __len__(self):
        return len(self.image_label_pairs)


import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGQCNN(nn.Module):
    def __init__(self):
        super(SimpleGQCNN, self).__init__()
        # Image stream
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # Calculate the output size after convolutions and pooling
        self.fc3_input_size = self._get_conv_output_size((1, 32, 32))
        self.fc3 = nn.Linear(self.fc3_input_size, 1024)

        # Merge stream
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1)  # Output: single value (e.g., quality score)

    def _get_conv_output_size(self, shape):
        # Generate a dummy tensor to get the output size after conv layers
        dummy_input = torch.zeros(1, *shape)
        with torch.no_grad():
            x = F.relu(self.conv1_1(dummy_input))
            x = F.max_pool2d(x, kernel_size=1, stride=1)
            if True:
                x = F.local_response_norm(x, size=2)

            x = F.relu(self.conv1_2(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            if True:
                x = F.local_response_norm(x, size=2)

            x = F.relu(self.conv2_1(x))
            x = F.max_pool2d(x, kernel_size=1, stride=1)

            x = F.relu(self.conv2_2(x))
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            if True:
                x = F.local_response_norm(x, size=2)

        return x.numel()

    def forward(self, x_image):
        x = F.relu(self.conv1_1(x_image))
        x = F.max_pool2d(x, kernel_size=1, stride=1)
        if True:
            x = F.local_response_norm(x, size=2)

        x = F.relu(self.conv1_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        if True:
            x = F.local_response_norm(x, size=2)

        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(x, kernel_size=1, stride=1)

        x = F.relu(self.conv2_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        if True:
            x = F.local_response_norm(x, size=2)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc3(x))

        x = F.relu(self.fc4(x))
        x = self.fc5(
            x
        )  # Output layer without activation function for regression output

        return x


def init_model():
    model = SimpleGQCNN()
    model = model.to("cuda")

    criterion = torch.nn.HuberLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    return model, criterion, optimizer, scheduler


def train(train_dataloader, val_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epoch = 100

    model, criterion, optimizer, scheduler = init_model()
    model.state_dict = torch.load(
        "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/Depth-Anything/Baselines/DexNet3.0/checkpoint_dexnet.pth"
    )

    # progress_bar = tqdm(range(num_epoch*len(train_dataloader) + num_epoch*len(val_dataloader)))

    running_loss = 0.0

    scaling_factor = 10e4
    gamma = 2

    model.train()

    lowest_loss = 10000.0

    for epoch in range(num_epoch):
        running_loss = 0.0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            images = images.squeeze(0).unsqueeze(1)
            labels = labels.squeeze(0).unsqueeze(1)
            optimizer.zero_grad()

            score = model(images)
            loss = criterion(score, labels)
            scaled_loss = loss * scaling_factor

            scaled_loss.backward()
            optimizer.step()
            running_loss += scaled_loss.item()

            # progress_bar.update(1)
            # print(f"Loss: {loss.item()}")

        print(torch.max(score), torch.min(score), torch.max(labels), torch.min(labels))
        epoch_loss = running_loss / (len(train_dataloader) * 64)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

        model.eval()  # Set the model to evaluation mode
        validation_loss = 0.0

        with torch.no_grad():  # Disable gradient computation
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)

                images = images.squeeze(0).unsqueeze(1)
                labels = labels.squeeze(0).unsqueeze(1)
                score = model(images)
                loss = criterion(score, labels)
                scaled_loss = loss * scaling_factor

                validation_loss += scaled_loss.item()

                # progress_bar.update(1)
        avg_validation_loss = validation_loss / (len(val_dataloader) * 64)
        if avg_validation_loss < lowest_loss:
            lowest_loss = avg_validation_loss
            torch.save(
                model,
                "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/Depth-Anything/Baselines/DexNet3.0/checkpoint_dexnet.pth",
            )
        print(f"Validation Loss: {avg_validation_loss}")

        model.train()

        scheduler.step(avg_validation_loss)


if __name__ == "__main__":
    dataset = CustomDataset(
        images_root="/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/compress_observation",
        labels_root="/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/labels_nbv_3F",
    )

    print(len(dataset))
    # Determine sizes for training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=8
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Example training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(train_dataloader, val_dataloader)
