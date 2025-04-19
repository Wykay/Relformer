# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Jialian Wu from https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/visualizer.py
import torch
import numpy as np
# from detectron2.structures import  BoxMode
from detectron2.structures.boxes import Boxes,BoxMode

from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from detectron2.structures import Boxes, RotatedBoxes, pairwise_iou, pairwise_iou_rotated
import os

class Visualizer_GRiT(Visualizer):
    def __init__(self, image, instance_mode=None):
        super().__init__(image, instance_mode=instance_mode)

    def draw_instance_predictions(self, predictions):
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        object_description = predictions.pred_object_descriptions.data
        # uncomment to output scores in visualized images
        # object_description = [c + '|' + str(round(s.item(), 1)) for c, s in zip(object_description, scores)]

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        ious = pairwise_iou(boxes,boxes)

        # np_boxes = []
        # for box in boxes:
        #     np_boxes.append(box.numpy())
        # combined_list = sorted(zip(scores.numpy(),boxes,classes,object_description))
        # scores, boxes, classes,object_description = zip(*combined_list)
        # boxes = [box.numpy() for box in boxes]
        # boxes = Boxes(boxes) 

        
        # scores_str = []
        # for i,score in enumerate(scores):
        #     scores_str.append(str(np.round(score,3)))

        # self.overlay_instances(
        #     masks=None,
        #     boxes=boxes,
        #     labels=object_description,
        #     # labels=scores_str,
        #     keypoints=None,
        #     assigned_colors=None,           # colors
        #     alpha=alpha,
        # )

        self.overlay_instances(
            masks=None,
            boxes=boxes,
            labels=object_description,
            keypoints=None,
            assigned_colors=None,
            alpha=alpha,
            ious=ious,
        )
        return self.output


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        predictions = self.predictor(image)                                             # __call__() missing 1 required positional argument: 'img_path'    __call__(self, original_image, img_path):  Tuple[Dict[str, torch.Tensor]]
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        
        # for CLIP's
        # image = image[:, :, ::-1]
        if type(image) == dict:
            image = image['image'][:, :, ::-1]

        else:
            image = image[:, :, ::-1]

        visualizer = Visualizer_GRiT(image, instance_mode=self.instance_mode)
        instances = predictions["instances"].to(self.cpu_device)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output