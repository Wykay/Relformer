from .modeling.meta_arch import grit,clip_meta_arch
from .modeling.roi_heads import grit_roi_heads,object_relation_roi_heads,clip_roi_heads,clip_obj_relation_roi_heads
from .modeling.backbone import vit
from .modeling.backbone import ria
from .modeling.roi_heads import mplug_roi_heads,whole2local_roi_heads,obj_rela_roi_heads_with_cross,rela_debug,relformer_clip_encode_region

from .data.datasets import object365
from .data.datasets import vg,vg2,vg_coco
from .data.datasets import grit_coco
