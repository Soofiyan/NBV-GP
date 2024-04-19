import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import matplotlib.pyplot as plt

model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14")
model = model.to("cuda")

# model.load_state_dict(
#     torch.load(
#         "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/Depth-Anything/huggingface/Depth-Anything/temp.pth"
#     )
# )

transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)

transform_affordance = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
    ]
)

url = "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/observation_data_nbv_3F/run_ID_0/3F/0/rgb_0_1.npy"
numpy_image = np.load(url)
crop_bin_dim = [73, 315, 195, 442]
numpy_image = numpy_image[
    crop_bin_dim[0] : crop_bin_dim[1], crop_bin_dim[2] : crop_bin_dim[3]
]
image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(numpy_image)
image = np.array(image) / 255.0
image = transform({"image": image})["image"]
image = torch.from_numpy(image).unsqueeze(0).to("cuda")


for name, param in model.named_parameters():
    if (
        "pretrained" in name
        or "depth_head.scratch.output_conv" in name
        or "depth_head.scratch.affordance" in name
    ):
        param.requires_grad = False
    # print(name, param.requires_grad)


# Step 3: Create the optimizer with only the binary classification parameters
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)

# Proceed with training as before
model.train()
optimizer.zero_grad()

image = image.to("cuda")


crop_bin_dim = [73, 315, 195, 442]
segmask = np.load(
    "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/observation_data_nbv_3F/run_ID_0/3F/0/segmask_0_1.npy",
).astype(np.uint8)
segmask = segmask[crop_bin_dim[0] : crop_bin_dim[1], crop_bin_dim[2] : crop_bin_dim[3]]
segmask = transform_affordance({"image": segmask})["image"]
mask = torch.from_numpy(segmask).unsqueeze(0).unsqueeze(1).to("cuda")

# Assuming `image` is your input tensor and `labels` is your target tensor
outputs, affordance_map = model(
    image, mask
)  # Adjusted to unpack the tuple returned by model
# Continue with your training loop using binary_probs for loss calculation
criterion = torch.nn.CrossEntropyLoss()

affordance_target = cv2.imread(
    "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/labels_nbv_3F/run_ID_0/3F/0/affordance_map_1_0_1.png",
    cv2.IMREAD_GRAYSCALE,
)

affordance_target = np.array(affordance_target) / 255.0
affordance_target = transform_affordance({"image": affordance_target})["image"]
affordance_target = (
    torch.from_numpy(affordance_target).unsqueeze(0).type(torch.float32).to("cuda")
)

segmask = np.load(
    "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/observation_data_nbv_3F/run_ID_0/3F/0/segmask_0_1.npy",
).astype(np.uint8)
segmask = segmask[crop_bin_dim[0] : crop_bin_dim[1], crop_bin_dim[2] : crop_bin_dim[3]]
segmask = transform_affordance({"image": segmask})["image"]
segmask = torch.from_numpy(segmask).unsqueeze(0).to("cuda")


# masked_affordance_map = affordance_map * segmask
# masked_affordance_target = affordance_target * segmask
# loss = F.mse_loss(masked_affordance_map, masked_affordance_target, reduction='sum') / segmask.sum()

loss = criterion(affordance_map, affordance_target).to("cuda")

print("loss", loss)
loss.backward()
optimizer.step()

model_path = "temp.pth"
torch.save(model.state_dict(), model_path)

model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    depth, affordance_map = model(image, mask)
    loss = criterion(affordance_map, affordance_target)
    print(loss)
    depth = depth.squeeze().detach().cpu().numpy()
    affordance_map = affordance_map.squeeze().detach().cpu().numpy()
    plt.imshow(affordance_map)
    plt.show()
