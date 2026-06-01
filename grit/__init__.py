"""Relformer / GRiT package init.

Importing the modules below registers their classes with Detectron2's
META_ARCH_REGISTRY / ROI_HEADS_REGISTRY / BACKBONE_REGISTRY, which is what
makes them resolvable from YAML configs by name. Adding a new component
requires importing it here.
"""

# Meta architectures
from .modeling.meta_arch import grit, clip_meta_arch

# ROI heads
from .modeling.roi_heads import grit_roi_heads, clip_obj_relation_roi_heads

# Backbones
from .modeling.backbone import vit

# Datasets — register splits at import time
from .data.datasets import object365
from .data.datasets import vg, vg2, vg_coco
from .data.datasets import grit_coco
