from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .data import IMAGENET_MEAN, IMAGENET_STD

BACKBONES = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V1),
}


class ResNetClassifier(nn.Module):
    """A torchvision ResNet with the ImageNet head replaced by a species head."""

    mean, std = IMAGENET_MEAN, IMAGENET_STD   # input normalisation it was trained with

    def __init__(self, n_classes: int, backbone: str = "resnet50",
                 pretrained: bool = True, dropout: float = 0.0, freeze_bn: bool = False):
        """Build the backbone and attach a fresh head.

        Args:
            n_classes: number of species, the output dimension.
            backbone: one of `BACKBONES` -- resnet18, resnet34 or resnet50.
            pretrained: start from ImageNet weights rather than from scratch.
            dropout: dropout probability before the head. 0 disables it.
            freeze_bn: keep BatchNorm running statistics fixed during training.

        Raises:
            ValueError: if `backbone` is not in `BACKBONES`.
        """
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}, have {list(BACKBONES)}")
        ctor, weights = BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        self.d_feat = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.head = nn.Linear(self.d_feat, n_classes)
        self.freeze_bn = freeze_bn

    def train(self, mode: bool = True):
        """Set training mode, keeping BatchNorm in eval mode if `freeze_bn`.

        Args:
            mode: True for training, False for evaluation.

        Returns:
            self, as `nn.Module.train` does.
        """
        super().train(mode)
        if mode and self.freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The representation an alignment or adversarial loss should read.

        Args:
            x: image batch, `(batch, 3, size, size)`.

        Returns:
            Pooled pre-head features, `(batch, d_feat)`.
        """
        return self.backbone(x)

    def logits(self, feats: torch.Tensor) -> torch.Tensor:
        """Class scores from features you have already computed.

        Use this when you need both the representation and the prediction, so the
        backbone runs once:

            feats  = model.features(x)
            logits = model.logits(feats)

        Calling `model(x)` and `model.features(x)` on the same batch runs the backbone
        twice and updates BatchNorm statistics twice per step. It still trains, which is
        what makes the mistake easy to miss.

        Args:
            feats: output of `features()`, `(batch, d_feat)`.

        Returns:
            Class scores, `(batch, n_classes)`.
        """
        return self.head(self.dropout(feats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images.

        Args:
            x: image batch, `(batch, 3, size, size)`.

        Returns:
            Class scores, `(batch, n_classes)`.
        """
        return self.logits(self.features(x))


# CLIP image towers in timm, and the open_clip model holding the matching text tower.
CLIP_TEXT = {"vit_base_patch16_clip_quickgelu_224.openai": ("ViT-B-16-quickgelu", "openai"),
             "vit_large_patch14_clip_quickgelu_224.openai": ("ViT-L-14-quickgelu", "openai")}
# The iWildCam prompts of WiSE-FT and FLYP (src/templates/iwildcam_template.py).
CLIP_TEMPLATES = ("a photo of {}.", "{} in the wild.")


@torch.no_grad()
def clip_zeroshot_weights(backbone: str, classnames: list[str]) -> torch.Tensor:
    """Zero-shot head weights for a CLIP backbone, as WiSE-FT and FLYP initialise it.

    Each class is the normalised mean of its normalised prompt embeddings, scaled by
    CLIP's learned logit scale (about 100), so `head(normalised image features)` starts
    out as CLIP's zero-shot classifier.

    Args:
        backbone: a key of `CLIP_TEXT`.
        classnames: one English name per class, in label order.

    Returns:
        `(n_classes, embed_dim)` weight matrix.
    """
    import open_clip
    arch, tag = CLIP_TEXT[backbone]
    clip = open_clip.create_model(arch, pretrained=tag).eval()
    tokenizer = open_clip.get_tokenizer(arch)
    rows = []
    for name in classnames:
        emb = F.normalize(clip.encode_text(tokenizer([t.format(name) for t in CLIP_TEMPLATES])),
                          dim=-1)
        rows.append(F.normalize(emb.mean(0), dim=0))
    return clip.logit_scale.exp() * torch.stack(rows)


class TimmClassifier(nn.Module):
    """A timm backbone (CLIP, DINOv2, ConvNeXt, ...) with a species head.

    Same interface as `ResNetClassifier`. The head is the one timm builds for the
    backbone -- so backbone-specific head init, e.g. ConvNeXt's `head_init_scale`, applies
    -- and `mean`/`std` are the checkpoint's own normalisation constants.
    """

    def __init__(self, n_classes: int, backbone: str, pretrained: bool = True,
                 dropout: float = 0.0, zeroshot_head: bool = False,
                 classnames: list[str] | None = None, **timm_kwargs):
        """Build the backbone and its head.

        Args:
            n_classes: number of species, the output dimension.
            backbone: a timm model name with its pretrained tag, e.g.
                `"vit_base_patch14_dinov2.lvd142m"`.
            pretrained: load the pretrained weights.
            dropout: dropout probability before the head. 0 disables it.
            zeroshot_head: CLIP only. Keep CLIP's image projection, L2-normalise the
                features and start the head from the zero-shot text classifier
                (WiSE-FT / FLYP). Needs `classnames`.
            classnames: English class names, for `zeroshot_head`.
            **timm_kwargs: passed to `timm.create_model`, e.g. `img_size`,
                `drop_path_rate`, `head_init_scale`.
        """
        super().__init__()
        import timm
        self.normalize = zeroshot_head
        if zeroshot_head:
            # The pretrained head of a timm CLIP tower is its 768 -> 512 projection; keep it.
            net = timm.create_model(backbone, pretrained=pretrained, **timm_kwargs)
            self.d_feat = net.num_classes
            self.head = nn.Linear(self.d_feat, n_classes)
            self.head.weight.data.copy_(clip_zeroshot_weights(backbone, classnames))
            self.head.bias.data.zero_()
        else:
            net = timm.create_model(backbone, pretrained=pretrained, num_classes=n_classes,
                                    **timm_kwargs)
            self.head = net.get_classifier()
            net.reset_classifier(0)          # the backbone now returns pooled features
            self.d_feat = net.num_features
        self.backbone = net
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.mean, self.std = net.pretrained_cfg["mean"], net.pretrained_cfg["std"]

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled pre-head features, `(batch, d_feat)`; L2-normalised for a zero-shot head."""
        f = self.backbone(x)
        return F.normalize(f, dim=-1) if self.normalize else f

    def logits(self, feats: torch.Tensor) -> torch.Tensor:
        """Class scores from `features()`, `(batch, n_classes)`."""
        return self.head(self.dropout(feats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images, `(batch, n_classes)`."""
        return self.logits(self.features(x))


def build_model(name: str, n_classes: int, **kw) -> nn.Module:
    """Build the model named in a config.

    Args:
        name: one of `BACKBONES` (a torchvision ResNet), or any timm model name with its
            pretrained tag, e.g. `"convnext_base.fb_in22k"`.
        n_classes: number of species.
        **kw: passed to `ResNetClassifier` -- `pretrained`, `dropout`, `freeze_bn` -- or
            to `TimmClassifier`.

    Returns:
        A `ResNetClassifier` or a `TimmClassifier`.
    """
    if name in BACKBONES:
        return ResNetClassifier(n_classes, backbone=name, **kw)
    return TimmClassifier(n_classes, backbone=name, **kw)
