import torch
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize
from PIL import Image
from zoedepth.utils.misc import save_raw_16bit

# ZoeD_N
overwrite = {
    "pretrained_resource": "local::./checkpoints/depth_anything_metric_depth_indoor.pt"
}
config = get_config("zoedepth", "infer", **overwrite)
model_zoe_n = build_model(config)


##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)


# Local file
image = Image.open(
    "/SATA_2TB_HD/Next-Best-View-3D-Object-Manipulation/Depth-Anything/metric_depth/data/nyu/basement_0001a/rgb_00000.jpg"
).convert(
    "RGB"
)  # load
depth = zoe.infer_pil(image)  # as numpy

fpath = "/SATA_2TB_HD/Next-Best-View-3D-Object-Manipulation/Depth-Anything/metric_depth/data/output.png"
save_raw_16bit(depth, fpath)

colored = colorize(depth)
Image.fromarray(colored).save(fpath)
