import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

import open3d as o3d

import torch
from torch.nn.functional import softmax
import torch.nn.functional as F

import glob
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import open3d as o3d

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()


def zero_pixels(data):
    zero_mask = torch.sum(data, dim=-1) == 0
    zero_px = torch.nonzero(zero_mask)
    return zero_px


def normal_cloud_project(rank, data, ksize=3):
    height = data.shape[0]
    width = data.shape[1]

    height, width = data.shape[:2]
    ksize = 3  # Example kernel size

    # Define Sobel kernels for x and y gradients
    sobel_x = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(rank)
    )
    sobel_y = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(rank)
    )

    # Assuming data is in shape [H, W, C], we need it in [B, C, H, W] format where B is batch size
    data = data.permute(2, 0, 1).unsqueeze(0)

    sobel_x = sobel_x.repeat(3, 1, 1, 1)  # Adjusting the kernel for 3 channels
    sobel_y = sobel_y.repeat(3, 1, 1, 1)  # Adjusting the kernel for 3 channels

    # Use groups parameter to apply convolution separately for each channel
    gx = F.conv2d(data, sobel_x, padding=1, groups=3)
    gy = F.conv2d(data, sobel_y, padding=1, groups=3)

    # Apply Sobel operator to compute gradients in x and y directions
    # gx = F.conv2d(data, sobel_x, padding=1)
    # gy = F.conv2d(data, sobel_y, padding=1)

    # Reshape gradients to [height*width, 3]
    gx_data = gx.squeeze(0).permute(1, 2, 0).reshape(height * width, 3)
    gy_data = gy.squeeze(0).permute(1, 2, 0).reshape(height * width, 3)

    # Compute cross product
    pc_grads = torch.cross(gx_data, gy_data)

    # normalize
    pc_grad_norms = torch.linalg.norm(
        pc_grads, dim=1, keepdim=True
    )  # Compute norms along the last dimension
    mask = pc_grad_norms.squeeze() > 0  # Identify non-zero norms

    # For non-zero norms, normalize the gradients
    pc_grads[mask] = pc_grads[mask] / pc_grad_norms[mask]

    # For zero norms, set to [0, 0, -1]
    zero_norm_mask = pc_grad_norms.squeeze() == 0
    pc_grads[zero_norm_mask] = torch.tensor([0, 0, -1.0], device=pc_grads.device)

    # Reshape
    normal_im_data = pc_grads.reshape(height, width, 3).to(rank)

    # preserve zeros
    zero_px = zero_pixels(data).to(rank)
    normal_im_data[zero_px[:, 0], zero_px[:, 1], :] = torch.zeros(3).to(rank)

    return normal_im_data


def deproject(rank, depth_image):
    height = depth_image.shape[0]
    width = depth_image.shape[1]
    _K = torch.tensor(
        [
            [457.0074, 0, 123.5],
            [0, 457.0074, 121],
            [0, 0, 1],
        ]
    )

    height, width = depth_image.shape
    row_indices, col_indices = torch.meshgrid(
        torch.arange(height, device=rank),
        torch.arange(width, device=rank),
        indexing="ij",
    )
    pixels = torch.stack((col_indices.flatten(), row_indices.flatten()), dim=0)
    pixels_homog = torch.cat(
        (pixels, torch.ones(1, pixels.shape[1], device=rank)), dim=0
    ).to(rank)
    depth_arr = depth_image.flatten().repeat(3, 1).to(rank)
    _K_inv = torch.linalg.inv(_K).to(rank)
    points_3d = _K_inv @ (pixels_homog * depth_arr)
    return points_3d


def deproject_to_image(rank, depth_image):
    height = depth_image.shape[0]
    width = depth_image.shape[1]
    point_cloud = deproject(rank, depth_image)
    point_cloud_im_data = point_cloud.T.reshape(height, width, 3)
    return point_cloud_im_data


def rotation_matrix(pitch, roll, device="cuda", dtype=torch.float16):
    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    cos_r = torch.cos(roll)
    sin_r = torch.sin(roll)

    Rx = torch.tensor(
        [[1, 0, 0], [0, cos_p, -sin_p], [0, sin_p, cos_p]], device=device, dtype=dtype
    )
    Ry = torch.tensor(
        [[cos_r, 0, sin_r], [0, 1, 0], [-sin_r, 0, cos_r]], device=device, dtype=dtype
    )

    return torch.mm(Ry, Rx)


def update_depth(rotated_points, shape):
    # Assuming the camera is at the origin and looking along the z-axis
    new_depths = rotated_points[:, 2].view(shape[0], shape[1])
    return new_depths


def apply_rotation(point_cloud, normals, pitch, roll, device="cuda"):
    dtype = point_cloud.dtype  # Adapt to the input tensor's dtype
    R = rotation_matrix(pitch, roll, device=device, dtype=dtype)
    rotated_points = torch.einsum("ij,kj->ki", R, point_cloud)
    rotated_normals = torch.einsum("ij,kj->ki", R, normals)
    return rotated_points, rotated_normals


class CropTransform:
    def __init__(self, crop_dim):
        self.crop_dim = crop_dim

    def __call__(self, x):
        return x[
            self.crop_dim[0] : self.crop_dim[1], self.crop_dim[2] : self.crop_dim[3]
        ]


class CustomDataset(Dataset):
    def __init__(self, rank, images_root):
        self.images_root = images_root

        self.resize_transform = transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.image_label_pairs = self._get_image_label_pairs()

        self.pattern = r"affordance_map_(\d+)_"
        self.limit_angle = 31 * torch.pi / 180

        positive_range = torch.linspace(0, torch.pi / 6, 7)
        negative_range = torch.linspace(-torch.pi / 6, 0, 7)
        self.pitch_angle = torch.concatenate(
            (
                positive_range,
                negative_range[:-1],
                torch.tensor([-self.limit_angle, self.limit_angle]),
            )
        )
        self.roll_angle = torch.concatenate(
            (
                positive_range,
                negative_range[:-1],
                torch.tensor([-self.limit_angle, self.limit_angle]),
            )
        )
        print(self.pitch_angle * 180 / torch.pi)
        print(self.roll_angle * 180 / torch.pi)

        self.rank = rank

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

                    segmask_file = image_file.replace("rgb", "segmask").replace(
                        ".png", ".npy"
                    )
                    depth_image_file = (
                        image_file.replace("rgb", "depth_image")
                        .replace(".jpg", ".npy")
                        .replace("compress_observation", "observation_data_nbv_3F")
                    )
                    pairs.append((image_file, depth_image_file, segmask_file))
        return pairs

    def __getitem__(self, idx):
        image_file, depth_image_file, segmask_file, view_number = (
            self.image_label_pairs[idx]
        )
        # print(image_file, depth_image_file, segmask_file, view_number)

        crop_bin_dim = [73, 287, 195, 442]

        # Load image
        # numpy_image = cv2.imread(image_file).astype(np.uint8)

        # numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
        # numpy_image = numpy_image[
        #     crop_bin_dim[0] : crop_bin_dim[1], crop_bin_dim[2] : crop_bin_dim[3]
        # ]
        # rgb_image_cropped = torch.from_numpy(numpy_image).cuda()

        depth_image = np.load(depth_image_file)
        depth_image = depth_image[
            crop_bin_dim[0] : crop_bin_dim[1], crop_bin_dim[2] : crop_bin_dim[3]
        ]
        depth_image_cropped = torch.from_numpy(depth_image).to(torch.float16).cuda()

        numpy_mask = cv2.imread(segmask_file, cv2.IMREAD_UNCHANGED)
        numpy_mask = numpy_mask[
            crop_bin_dim[0] : crop_bin_dim[1], crop_bin_dim[2] : crop_bin_dim[3]
        ]
        segmask_cropped = torch.from_numpy(numpy_mask).cuda()

        affordance_map_path = ""
        affordance_map_pitch_path = ""
        affordance_map_roll_path = ""

        mask_numbers = torch.unique(segmask_cropped)

        padding_size = [50, 50, 50, 50]
        segmask_cropped = F.pad(segmask_cropped, padding_size, "constant", 0)
        depth_image_cropped = F.pad(depth_image_cropped, padding_size, "constant", 0)

        temp_path = image_file.replace("compress_observation", "labels_nbv_3F_angle_c")

        os.makedirs(os.path.dirname(temp_path), exist_ok=True)

        if mask_numbers.shape[0] > 1:

            angle_resolution = 1
            angles = (
                torch.arange(0, 360, angle_resolution, device=self.rank).to(
                    torch.float16
                )
                * torch.pi
                / 180
            )
            cos_values = torch.cos(angles)
            sin_values = torch.sin(angles)

            for mask_number in mask_numbers:
                if mask_number == 0:
                    continue
                mask_indices = segmask_cropped == mask_number
                depth_image_masked = torch.where(
                    mask_indices,
                    depth_image_cropped,
                    torch.tensor(0).to(torch.float16).to(self.rank),
                )

                depth_image_normal = depth_image_masked

                point_cloud_im = deproject_to_image(self.rank, depth_image_normal).to(
                    self.rank
                )
                normal_cloud_im = normal_cloud_project(self.rank, point_cloud_im).to(
                    self.rank
                )

                initial_shape = point_cloud_im.shape

                point_cloud_flat = torch.reshape(point_cloud_im, (-1, 3)).to(self.rank)
                normal_cloud_flat = torch.reshape(normal_cloud_im, (-1, 3)).to(
                    self.rank
                )

                best_score = None
                angle_roll = None
                angle_pitch = None
                flag_first_angle = True

                affordance_map_path = (
                    image_file.replace("rgb", f"affordance_map_{mask_number}")
                    .replace(".jpg", ".png")
                    .replace("compress_observation", "labels_nbv_3F_angle_c")
                )
                affordance_map_roll_path = (
                    image_file.replace("rgb", f"affordance_map_roll_{mask_number}")
                    .replace(".jpg", ".png")
                    .replace("compress_observation", "labels_nbv_3F_angle_c")
                )
                affordance_map_pitch_path = (
                    image_file.replace("rgb", f"affordance_map_pitch_{mask_number}")
                    .replace(".jpg", ".png")
                    .replace("compress_observation", "labels_nbv_3F_angle_c")
                )

                if (
                    os.path.exists(affordance_map_path)
                    and os.path.exists(affordance_map_roll_path)
                    and os.path.exists(affordance_map_pitch_path)
                ):
                    continue

                for pitch in self.pitch_angle:
                    for roll in self.roll_angle:
                        point_cloud_im_rotated, normal_cloud_im_rotated = (
                            apply_rotation(
                                point_cloud_flat,
                                normal_cloud_flat,
                                pitch,
                                roll,
                                device=self.rank,
                            )
                        )
                        depth_image_masked_rotated = update_depth(
                            point_cloud_im_rotated, initial_shape
                        )

                        point_cloud_im_rotated = point_cloud_im_rotated.view(
                            initial_shape[0], initial_shape[1], 3
                        )
                        normal_cloud_im_rotated = normal_cloud_im_rotated.view(
                            initial_shape[0], initial_shape[1], 3
                        )

                        # plt.imshow(new_depths.cpu().detach().numpy())
                        # plt.show()

                        affordance_map = torch.zeros_like(segmask_cropped)
                        affordance_map_roll = torch.zeros_like(segmask_cropped)
                        affordance_map_pitch = torch.zeros_like(segmask_cropped)

                        non_zero_indices = torch.nonzero(depth_image_masked_rotated)
                        non_zero_coordinates = non_zero_indices[:, [1, 0]]
                        non_zero_coordinates_list = list(
                            map(tuple, non_zero_coordinates.tolist())
                        )

                        non_zero_coordinates = torch.tensor(
                            non_zero_coordinates_list, device=self.rank
                        ).long()

                        # Compute base coordinates for all non-zero coordinates
                        base_coordinates = non_zero_coordinates

                        normal_values_point_pixel = normal_cloud_im_rotated[
                            base_coordinates[:, 1], base_coordinates[:, 0]
                        ]
                        depth_values_point_pixel = depth_image_masked_rotated[
                            base_coordinates[:, 1], base_coordinates[:, 0]
                        ]

                        clamped_depth = torch.clamp(
                            depth_values_point_pixel, min=0.4, max=0.6
                        )
                        lower_pixel_to_m = 12
                        upper_pixel_to_m = 17
                        suction_cup_size = (
                            (
                                ((clamped_depth - 0.6) / (0.6 - 0.4))
                                * (lower_pixel_to_m - upper_pixel_to_m)
                                + lower_pixel_to_m
                            )
                            .round()
                            .long()
                        )

                        suction_cup_size = (suction_cup_size).unsqueeze(1)
                        suction_cup_size = suction_cup_size.expand(-1, angles.shape[0])

                        # Compute offsets for all points in a single operation
                        x_offsets = suction_cup_size * cos_values.unsqueeze(0)
                        y_offsets = suction_cup_size * sin_values.unsqueeze(0)

                        # Calculate new coordinates based on offsets
                        x_coords = base_coordinates[:, 0].unsqueeze(1) + x_offsets
                        y_coords = base_coordinates[:, 1].unsqueeze(1) + y_offsets

                        # Round and convert to long for indexing
                        x_coords_int = x_coords.round().long()
                        y_coords_int = y_coords.round().long()
                        # print(x_coords_int[0], y_coords_int[0])

                        # Extract depth and normal values in bulk
                        depth_values = depth_image_masked_rotated[
                            y_coords_int, x_coords_int
                        ]
                        normal_values = normal_cloud_im_rotated[
                            y_coords_int, x_coords_int
                        ]

                        # Compute the reference normal and scores
                        reference_normals = torch.mean(normal_values, dim=1)
                        reference_normals = reference_normals / torch.norm(
                            reference_normals, dim=1
                        ).unsqueeze(1)
                        dot_products = torch.einsum(
                            "ijk,ik->ij", normal_values, reference_normals
                        )
                        angles_between_vectors = torch.acos(
                            torch.clamp(dot_products, -1.0, 1.0)
                        )
                        angle_threshold = torch.empty(
                            (angles_between_vectors.shape[0], 1), device=self.rank
                        ).fill_(torch.deg2rad(torch.tensor(45, device=self.rank)))
                        scores = torch.maximum(
                            torch.tensor(0).to(self.rank),
                            angle_threshold - angles_between_vectors,
                        )
                        normalized_scores = scores / angle_threshold * 100
                        normal_consistency_score = torch.mean(normalized_scores, dim=1)

                        # Angle inclined calculation
                        angle_inclined = (
                            (
                                torch.abs(normal_values_point_pixel[:, 0])
                                + torch.abs(normal_values_point_pixel[:, 1])
                            )
                            * 180
                            / torch.pi
                        )
                        angle_inclined_score = (
                            (torch.tensor(75).to(self.rank) - angle_inclined) * 100 / 75
                        )

                        # Depth variability
                        depth_variability = torch.std(depth_values, dim=1)

                        # Cost calculation
                        num_measurements = depth_values.shape[1]
                        batch_size = depth_values.shape[0]
                        # Spacing calculation assuming angle_resolution is defined
                        spacing = torch.tensor(int(90 / angle_resolution)).to(self.rank)
                        indices = (
                            torch.arange(num_measurements).unsqueeze(0).to(self.rank)
                            + spacing
                        ) % num_measurements
                        indices_repeated = indices.expand(batch_size, -1)
                        differences = torch.abs(
                            depth_values
                            - torch.gather(depth_values, 1, indices_repeated)
                        )
                        cost = 10 * angle_resolution * differences.sum(dim=1)

                        # Anomaly scores and normalization
                        max_anomaly_score = torch.tensor(20.0).to(self.rank)
                        expected_depth, _ = torch.max(depth_values, dim=1, keepdim=True)
                        anomaly_scores = expected_depth - depth_values
                        sum_of_anomaly_scores = torch.sum(anomaly_scores, dim=1)
                        normalized_anomaly_score = (
                            1 - sum_of_anomaly_scores / max_anomaly_score
                        ) * 100

                        k1, k2, k3, k4, k5 = [0.2] * 5

                        final_scores = (
                            normalized_anomaly_score * k1
                            + (100 - cost) * k2
                            + (100 - depth_variability * 10e3) * k3
                            + normal_consistency_score * k4
                            + angle_inclined_score * k5
                        )

                        final_scores = torch.clamp(final_scores, min=0)
                        if flag_first_angle:
                            best_score = torch.zeros_like(final_scores)
                            angle_roll = torch.zeros_like(final_scores)
                            angle_pitch = torch.zeros_like(final_scores)
                            flag_first_angle = False

                        greater = final_scores > best_score

                        # best_score = torch.max(best_score, final_scores)
                        best_score[greater] = final_scores[greater]

                        if roll == self.limit_angle or roll == -self.limit_angle:
                            encoded_roll = (
                                (torch.tensor(0.0) * 180 / torch.pi + 30) / 60
                            ) * 255
                        else:
                            encoded_roll = ((roll * 180 / torch.pi + 30) / 60) * 255

                        if pitch == self.limit_angle or pitch == -self.limit_angle:
                            encoded_pitch = (
                                (torch.tensor(0.0) * 180 / torch.pi + 30) / 60
                            ) * 255
                        else:
                            encoded_pitch = ((pitch * 180 / torch.pi + 30) / 60) * 255

                        angle_roll[greater] = encoded_roll.to(torch.float32).to(
                            self.rank
                        )
                        angle_pitch[greater] = encoded_pitch.to(torch.float32).to(
                            self.rank
                        )

                # num_zero_values_per_batch = torch.count_nonzero(
                #     depth_values == 0, axis=1
                # )
                # total_elements_per_batch = depth_values.shape[1]
                # percentage_zero_per_batch = (
                #     num_zero_values_per_batch / total_elements_per_batch
                # )
                # batches_with_20p_zeros = percentage_zero_per_batch >= 0.2
                # best_score[batches_with_20p_zeros] = 0

                # angle_roll[batches_with_20p_zeros] = 0
                # angle_pitch[batches_with_20p_zeros] = 0
                base_coordinates = base_coordinates.to(torch.int)

                affordance_map[base_coordinates[:, 1], base_coordinates[:, 0]] = (
                    (best_score * 255 / 100).round().to(torch.int)
                )
                affordance_map_roll[base_coordinates[:, 1], base_coordinates[:, 0]] = (
                    (angle_roll).round().to(torch.int)
                )
                affordance_map_pitch[base_coordinates[:, 1], base_coordinates[:, 0]] = (
                    (angle_pitch).round().to(torch.int)
                )

                affordance_map = affordance_map[
                    padding_size[0] : -padding_size[1],
                    padding_size[2] : -padding_size[3],
                ]
                affordance_map_roll = affordance_map_roll[
                    padding_size[0] : -padding_size[1],
                    padding_size[2] : -padding_size[3],
                ]
                affordance_map_pitch = affordance_map_pitch[
                    padding_size[0] : -padding_size[1],
                    padding_size[2] : -padding_size[3],
                ]

                cv2.imwrite(
                    affordance_map_path, affordance_map.cpu().numpy().astype(np.uint8)
                )
                cv2.imwrite(
                    affordance_map_roll_path,
                    affordance_map_roll.cpu().numpy().astype(np.uint8),
                )
                cv2.imwrite(
                    affordance_map_pitch_path,
                    affordance_map_pitch.cpu().numpy().astype(np.uint8),
                )

        return affordance_map_path, affordance_map_roll_path, affordance_map_pitch_path

    def __len__(self):
        return len(self.image_label_pairs)


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def print_memory_usage(rank):
    print(f"Rank {rank}: Allocated memory: {torch.cuda.memory_allocated(rank)} bytes")
    print(f"Rank {rank}: Reserved memory: {torch.cuda.memory_reserved(rank)} bytes")


import gc


def collect_garbage():
    gc.collect()
    torch.cuda.empty_cache()


def collect_data(rank, world_size, images_root_path):
    setup(rank, world_size)

    dataset = CustomDataset(rank, images_root=images_root_path)

    print(f"Rank {rank}: Dataset length: {len(dataset)}")

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    process_dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=18,
        pin_memory=False,
    )

    count = 0
    for (
        affordance_map_path,
        affordance_map_roll_path,
        affordance_map_pitch_path,
    ) in process_dataloader:
        print(count)
        count += 1

        # print_memory_usage(rank)

        # Explicitly delete or move tensors out of scope to free memory
        del affordance_map_path, affordance_map_roll_path, affordance_map_pitch_path

        # Collect garbage
        collect_garbage()

    cleanup()


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    world_size = torch.cuda.device_count()

    images_root = "/home/ubuntu/compress_observation"

    mp.spawn(collect_data, args=(world_size, images_root), nprocs=world_size, join=True)
