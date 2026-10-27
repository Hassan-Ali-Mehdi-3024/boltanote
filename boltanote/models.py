"""The three models BoltaNote compares, smallest-to-phone-friendliest.

Every model takes RGB photos as floats in [0, 1], shape [B, 3, 224, 224] (the whole camera frame
squeezed to a square), does its own normalisation, and returns logits for 15 classes: each of the
7 notes front and back, plus "none" (no note in view). Front and back are separate classes because
they look nothing alike; the app adds their probabilities to get one per note value.

  cnn_scratch              a 5-block CNN from random weights (the baseline)
  mobilenetv3_large        MobileNetV3-Large (Howard et al. 2019), ImageNet-pretrained, 5.5M params
  mobilenetv4_conv_medium  MobileNetV4-Conv-Medium (Qin et al. 2024), ImageNet-pretrained, 9.7M params
"""

import torch
import torch.nn as nn

VALUES = ["10", "20", "50", "100", "500", "1000", "5000"]
CLASSES = [f"{v}_{s}" for v in VALUES for s in ("front", "back")] + ["none"]
GROUPS = VALUES + ["none"]  # what the app reports
GROUP_OF = [GROUPS.index(c.split("_")[0]) for c in CLASSES]  # class index -> group index

MODELS = ["cnn_scratch", "mobilenetv3_large", "mobilenetv4_conv_medium"]
TIMM = {
    "mobilenetv3_large": "mobilenetv3_large_100.ra_in1k",
    "mobilenetv4_conv_medium": "mobilenetv4_conv_medium.e500_r224_in1k",
}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Normalised(nn.Module):
    def __init__(self, net, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def _scratch(n_classes):
    def block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256), block(256, 256),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(256, n_classes))


def build_model(name, pretrained=False, n_classes=len(CLASSES)):
    if name == "cnn_scratch":
        return Normalised(_scratch(n_classes), mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
    if name in TIMM:
        import timm

        net = timm.create_model(TIMM[name], pretrained=pretrained, num_classes=n_classes, drop_rate=0.2)
        cfg = net.pretrained_cfg
        return Normalised(net, mean=cfg.get("mean", IMAGENET_MEAN), std=cfg.get("std", IMAGENET_STD))
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ck["model"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


class ExportWrapper(nn.Module):
    """What goes into the ONNX file: class probabilities, softmax included."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.softmax(self.model(x), 1)
