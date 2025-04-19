from typing import Dict, List, Optional, Tuple
# from sympy import false
import torch,yaml
from detectron2.config import configurable
from detectron2.structures import Instances
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from .mplug.model_caption_mplug import MPLUG
from transformers import AutoTokenizer

from .clip import clip
from PIL import Image
import copy


@META_ARCH_REGISTRY.register()
class GRiT_w_CLIP(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        **kwargs):
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
    
        config_path = 'configs/caption_mplug_large.yaml'
        config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        config["min_length"] = 8                                                       
        config["max_length"] = 25
        config["max_input_length"] = 25                                                 
        config["add_object"] = True
        config["beam_size"] = 1
        config['text_encoder'] = 'bert-base-uncased'
        config['text_decoder'] = 'bert-base-uncased'

        mplug_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

        mPLUG_model = MPLUG(config=config, tokenizer=mplug_tokenizer)  

        self.roi_heads.text_decoder = mPLUG_model.text_decoder               # 131.5 M
        self.clip_backbone, self.clip_preprocess = clip.load('ViT-B/16')

    @classmethod
    def from_config(cls, cfg):                                                                            
        ret = super().from_config(cfg)                                                                   
        return ret

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)                                      # 256 instances from proposals
   
        with torch.no_grad():
            cl_images = [Image.open(x["file_name"]) for x in batched_inputs]
            image_features = []
            for img in cl_images:
                image_features.append(self.clip_preprocess(img).unsqueeze(0).type(self.clip_backbone.dtype))
            clip_features = self.clip_backbone.visual(torch.cat(image_features,dim=0).to(self.device))                          # bs,197,768  _,0,_ is <cls>

        results, _ = self.roi_heads(features, proposals, clip_features=clip_features)                                                    # 
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return GRiT_w_CLIP._postprocess(
                results, batched_inputs, images.image_sizes)
        else:
            return results

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        targets_task = batched_inputs[0]['task']


        features = self.backbone(images.tensor)                     # dict: 'p3','p4','p5'  each bs,256,128,128
        proposals, proposal_losses = self.proposal_generator(       # proposals list: len=bs, 
            images, features, gt_instances)

        with torch.no_grad():
            images = [Image.open(x["file_name"]) for x in batched_inputs]
            image_features = []
            for img in images:
                image_features.append(self.clip_preprocess(img).unsqueeze(0).type(self.clip_backbone.dtype))
            clip_features = self.clip_backbone.visual(torch.cat(image_features,dim=0).to(self.device))                          # bs,197,768  _,0,_ is <cls>

        pred_instances, roihead_textdecoder_losses = self.roi_heads(
            features, proposals, gt_instances, targets_task, clip_features)

        losses = {}
        losses.update(roihead_textdecoder_losses)
        losses.update(proposal_losses)

        return losses
