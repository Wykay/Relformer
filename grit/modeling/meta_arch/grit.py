from typing import Dict, List, Optional, Tuple
# from sympy import false
import torch,yaml
from torch import nn
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from .mplug.model_caption_mplug import MPLUG
from .mplug.tokenization_bert import BertTokenizer
from .mplug.vit import resize_pos_embed
# from .mplug.modeling_mplug import BertPrefixModel
# from .mplug.predictor import TextGenerator

@META_ARCH_REGISTRY.register()
class GRiT(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        **kwargs):
        super().__init__(**kwargs)
        assert self.proposal_generator is not None
    
        # initialize mPLUG's text_decoder with checkpoints
        config_path = 'configs/caption_mplug_large.yaml'
        config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        config["min_length"] = 2                                                        # 8
        config["max_length"] = 30                                                       # num of tokens
        config["add_object"] = True
        config["beam_size"] = 1
        config['text_encoder'] = 'bert-base-uncased'
        config['text_decoder'] = 'bert-base-uncased'

        # initilize mPLUG-Large-V2 model
        self.config = config
        mplug_tokenizer = BertTokenizer.from_pretrained('/root/.cache/huggingface/bert-base-uncased', local_files_only=True)             
        mPLUG_model = MPLUG(config=config, tokenizer=mplug_tokenizer)                      

        self.roi_heads.text_decoder = mPLUG_model.text_decoder


    @classmethod
    def from_config(cls, cfg):                                                                              # cls: class 'grit.modeling.meta_arch.grit.GRiT'
        ret = super().from_config(cfg)                                                                      # build model from config file,  dict:   'backbone', 'proposal_generator', 'roi_heads', 'input_format', 'vis_period', 'pixel_mean', 'pixel_std'
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
        results, _ = self.roi_heads(features, proposals)                                                    # 
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return GRiT._postprocess(
                results, batched_inputs, images.image_sizes)
        else:
            return results

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        targets_task = batched_inputs[0]['task']
        for anno_per_image in batched_inputs:
            assert targets_task == anno_per_image['task']

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)
        proposals, roihead_textdecoder_losses = self.roi_heads(
            features, proposals, gt_instances, targets_task=targets_task)

        losses = {}
        losses.update(roihead_textdecoder_losses)
        losses.update(proposal_losses)

        return losses