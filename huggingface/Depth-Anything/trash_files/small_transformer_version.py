from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import cv2

url = "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/observation_data_nbv_3F/run_ID_0/3F/0/rgb_0_1.npy"
numpy_image = np.load(url)
image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(numpy_image)

plt.imshow(image)
plt.show()

model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
# model.load_state_dict(
#     torch.load(
#         "/SATA_2TB_HD/Next-Best-View-3D-Object-Manipulation/Depth-Anything/huggingface/Depth-Anything/temp.pth"
#     )
# )

image_processor = AutoImageProcessor.from_pretrained(
    "LiheYoung/depth-anything-small-hf"
)
# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

url = "/media/soofiyan_ws/ced8ad28-6b75-4190-8413-8b02bc00b409/Next-Best-View-3D-Object-Manipulation/System_Identification_Data/observation_data_nbv_3F/run_ID_0/3F/0/depth_image_0_1.npy"
numpy_image = np.load(url)
plt.figure()
plt.imshow(numpy_image, cmap="inferno")
plt.show()

plt.imshow(predicted_depth.squeeze(0).numpy(), cmap="inferno")
plt.show()

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# visualize the prediction
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)

plt.imshow(depth, cmap="inferno")
plt.show()


# binary_classification_params = list(model.backbone.parameters()) + list(
#     model.head.binary_classification_head.parameters()
# )
# optimizer = torch.optim.Adam(binary_classification_params, lr=1e-4)
# model.train()

# optimizer.zero_grad()

# # Forward pass
# outputs = model(**inputs)
# binary_logits = (
#     outputs.binary_probs
# )  # Assuming this is the binary classification output
# criterion = torch.nn.BCEWithLogitsLoss()
# # Calculate the loss

# print(binary_logits)
# loss = criterion(binary_logits, torch.tensor([[1.0]]))

# # Backward pass and optimize
# loss.backward()
# optimizer.step()

# model_path = "temp.pth"
# torch.save(model.state_dict(), model_path)
