import yaml
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import copy

from detectron2.config import configurable
from detectron2.structures import Boxes,Instances,pairwise_iou
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.cascade_rcnn import _ScaleGradient
from detectron2.utils.events import get_event_storage

from .grit_roi_heads import GRiTROIHeadsAndTextDecoder
from grit.data.custom_dataset_mapper import ObjDescription

from .mplug.predictor import TextGenerator

from transformers import BertConfig
# from memory_efficient_attention_pytorch import Attention as memory_efficient_atn
from nltk.translate import meteor
import copy
import numpy as np
from torchvision.ops import box_iou
from .nms_with_cap import batched_soft_nms_with_cap_score as my_nms

# for atn visualization
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import csv,os
# from visualizer import get_local
# get_local.activate()

from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import batched_nms

# nltk.download('wordnet')
BertLayerNorm = torch.nn.LayerNorm

class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # @get_local('attention_probs')
    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)           # bs,seq,dim                        # [18, 196, 768]
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))                       # 18,12,196,196
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)                                          # 27,12,196,196     bs,n_heads,seq,seq

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropout = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs_dropout, value_layer)                              # 18,12,196,64

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)                                    # 18,196,768
        return context_layer

class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.selfatn = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.selfatn(input_tensor, input_tensor, attention_mask)                                                                  # input_tensor: 24,196,768
        attention_output = self.output(self_output, input_tensor)
        return attention_output
    
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        self.input_linear = nn.Linear(input_dim,output_dim)
        self.input_embedder_dropout = nn.Dropout(0.1)
        self.enc_reduce_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        _x = self.input_embedder_dropout(self.input_linear(x))
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        # add & norm
        x = self.enc_reduce_norm(x + _x)                                    
        return x

class obj_relation_encoder(nn.Module):
    def __init__(self,bert_config):
        super().__init__()

        self.num_layers = 1
        self.self_attn = nn.ModuleList()
        self.ffn = nn.ModuleList()
        for i in range(self.num_layers):
            self.self_attn.append(BertSelfattLayer(bert_config))
            self.ffn.append(
                nn.ModuleList(
                [nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
                nn.Linear(bert_config.hidden_size, bert_config.hidden_size)]
                ) 
            )
    
    def forward(self,x):

        for i in range(self.num_layers):
            x = self.self_attn[i](x, None)      # (object_features.size()[:-1],dtype=torch.long).to(object_features.device) aleady added                # 16,196,768

            for l in self.ffn[i]:
                x = x + l(x)                    # maybe layernorm( x + l(x) )

        return x
        


@ROI_HEADS_REGISTRY.register()
class CatAndCat(GRiTROIHeadsAndTextDecoder):
    @configurable 
    def __init__(
        self,
        *,
        text_decoder_transformer,
        train_task: list,
        test_task: str,
        mult_proposal_score: bool = False,
        mask_weight: float = 1.0,
        object_feat_pooler=None,
        soft_nms_enabled=False,
        beam_size=1,
        **kwargs,
    ):
        # GRiT : backbone, roi_heads, tokenizer(do_lower_case), SmoothLabelCrossEntropyLoss
        super().__init__(
        text_decoder_transformer = text_decoder_transformer,
        train_task = train_task,
        test_task = test_task,
        mult_proposal_score = mult_proposal_score,
        mask_weight = mask_weight,
        object_feat_pooler=object_feat_pooler,                # OBJECT_FEAT_POOLER_RES = 14
        soft_nms_enabled=soft_nms_enabled,
        beam_size=beam_size,
        **kwargs,)
        
        # self.map_img_to_text = MLP(256,768,768,5)

        self.beam_generator = None

        # initialize mPLUG's text_decoder with checkpoints
        config_path = 'configs/caption_mplug_large.yaml'
        config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        config["min_length"] = 2
        config["max_length"] = 30
        config["add_object"] = True
        config["beam_size"] = 1
        config['text_encoder'] = 'bert-base-uncased'
        config['text_decoder'] = 'bert-base-uncased'

        # initilize mPLUG-Large-V2 model 
        self.mplug_config = config


        self.map_img_to_text = nn.Linear(1024,768)

        self.clip_feat_pooler = copy.deepcopy(object_feat_pooler)
        self.clip_feat_pooler.level_poolers[0].spatial_scale = 14/1024 
        self.clip_feat_pooler.level_poolers = nn.ModuleList([self.clip_feat_pooler.level_poolers[0]])

        bert_config = BertConfig.from_json_file(self.mplug_config['bert_config']) 
        self.obj_relation_encoder = obj_relation_encoder(bert_config)

        # # Object feature encoding
        self.visn_fc = nn.Linear(768, 768)
        self.visn_layer_norm = BertLayerNorm(768, eps=1e-12)

        # # Box position encoding
        self.box_fc = nn.Linear(4, 768)
        self.box_layer_norm = BertLayerNorm(768, eps=1e-12)
        self.box_pos_embed = nn.Embedding(1, 768)                           #  self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # self.dropout = nn.Dropout(0.1)

        self.global_mapping = nn.Linear(768,1024)


    def _forward_box(self, features, proposals, targets=None, task="ObjectDet", clip_features=None):                   # targets = gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        if self.training:
            proposals = self.check_if_all_background(proposals, targets, 0)
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [p.get('scores') for p in proposals]
            else:
                proposal_scores = [p.get('objectness_logits') for p in proposals]

        features = [features[f] for f in self.box_in_features]
        clip_features = clip_features[:,1:,:]                   # 2,197,768 -> 2,196,768  prefix or init
        # clip_features = self.map_img_to_text(clip_features)     # 4, 196, 768
        clip_features = clip_features.permute(0,2,1).unflatten(-1,(14,14))

        head_outputs = []
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        # refine boxes predictions in a cascade manner
        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes, image_sizes,
                    logits=[p.objectness_logits for p in proposals])
                if self.training:
                    proposals = self._match_and_label_boxes_GRiT(
                        proposals, k, targets)
                    proposals = self.check_if_all_background(proposals, targets, k)
            predictions = self._run_stage(features, proposals, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(
                (predictions[0], predictions[1]), proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        # for each gpu  captioning task
        if self.training: 

            text_decoder_loss = 0
            # modified 
            # Single image
            for i,proposal in enumerate(proposals):
                object_descriptions = []
                object_descriptions += proposal.gt_object_descriptions[proposal.foreground > 0].data

                if len(object_descriptions) > 0:
                    foreground = proposal.foreground  

                    object_descriptions = ObjDescription(object_descriptions).data
                    answer = self.tokenizer(object_descriptions, padding='longest', truncation=True, max_length=self.mplug_config['max_length'], return_tensors="pt").to(features[0].device) 

                    current_features = [features[0][i].unsqueeze(0), features[1][i].unsqueeze(0), features[2][i].unsqueeze(0)]

                    vit_object_features = self.object_feat_pooler(current_features, [proposal.proposal_boxes])[foreground > 0]                    # 512,256,14,14
                    clip_object_features = self.clip_feat_pooler([clip_features[i].unsqueeze(0)], [proposal.proposal_boxes])[foreground > 0]                      # 可能可以直接从 14*14 裁剪   pool from clip fmap    512,768,14,14

                    clip_object_features = clip_object_features.view(
                        clip_object_features.shape[0], clip_object_features.shape[1], -1).permute(0, 2, 1).contiguous()                 # 49,784,768
                    clip_object_features = clip_object_features.type_as(vit_object_features)

                    vit_object_features = vit_object_features.view(
                        vit_object_features.shape[0], vit_object_features.shape[1], -1).permute(0, 2, 1).contiguous()                 # 49,784,768
                    
                    global_feature = self.global_mapping(torch.mean(clip_object_features, -2, keepdim=True))                                               # 49, 1, 768
                
                    object_features = torch.cat([vit_object_features,clip_object_features],-1)

                    object_features = torch.cat([object_features, global_feature], -2)

                    object_features = self.map_img_to_text(object_features)

                    boxes = proposal.proposal_boxes[foreground > 0].tensor      
                    x = self.visn_fc(object_features)
                    x = self.visn_layer_norm(x)

                    y = self.box_fc(boxes)                                        #  self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)      
                    y_ = self.box_pos_embed.weight.repeat(y.shape[0], 1) 
                    y = y + y_
                    y = self.box_layer_norm(y)

                    object_features = torch.cat([x, y.unsqueeze(1)], 1)

                    object_features = _ScaleGradient.apply(object_features, 1.0 / self.num_cascade_stages)

                    object_mask = torch.ones(object_features.size()[:-1],dtype=torch.long).to(object_features.device) 
             
                    # obj2cap
                    per_img_txt_loss = self.text_decoder(       answer.input_ids,                            
                                                            attention_mask = answer.attention_mask,           # gt      264,12
                                                            encoder_hidden_states = object_features,          # 264,196,768
                                                            encoder_attention_mask = object_mask,             # 264,196
                                                            labels = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100),                # gt with mask input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)  
                                                            return_dict = True,
                                                            reduction = 'none'
                                                            ).loss                                            # CrossEntropyLoss()   
                    
                    del object_features,object_descriptions
                    
                    text_decoder_loss += per_img_txt_loss
                else:
                    text_decoder_loss += head_outputs[0][1][0].new_zeros([1])[0]
            
            text_decoder_loss = text_decoder_loss / len(proposals)
            losses = {}
            storage = get_event_storage()

            if torch.isnan(text_decoder_loss) :                 # not a number
                text_decoder_loss = torch.Tensor([0]).to(torch.float16).to(features[0].device)

            # RoI Head losses (For the proposal generator loss, please find it in grit.py)
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                        stage_losses = predictor.losses(
                            (predictions[0], predictions[1]), proposals)
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})

            # Text Decoder loss
            losses.update({'text_decoder_loss': text_decoder_loss})                                                       #DEBUG: 0.5 text loss        
            return losses
        
        # inference
        else:
            logits_per_stage = [(h[1][0],) for h in head_outputs]

            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]


            logits = [
                sum(list(logits_per_image)) * (1.0 / self.num_cascade_stages)
                for logits_per_image in zip(*logits_per_stage)
            ]
            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 for s, ps in zip(scores, proposal_scores)]
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(
                (predictions[0], predictions[1]), proposals)
            assert len(boxes) == 1

            pred_instances, _ = self.fast_rcnn_inference_GRiT(                                                          #  len(pred_instances[0]) = 25   pred_instances: bs, num_ins
                boxes,
                scores,
                logits,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
                self.soft_nms_enabled,
                # False,
            )                                                                                                           # 1024 -> test_topk_per_image boxes  we do not do nms here, only using score_thresh to reduce proposals   

            assert len(pred_instances) == 1, "Only support one image"
            for i, pred_instance in enumerate(pred_instances):
                if len(pred_instance.pred_boxes) > 0:

                    current_features = [features[0][i].unsqueeze(0), features[1][i].unsqueeze(0), features[2][i].unsqueeze(0)]
                    vit_object_features = self.object_feat_pooler(current_features, [pred_instance.pred_boxes]).to(current_features[0].device)                                 # 512,256,14,14
                    clip_object_features = self.clip_feat_pooler([clip_features[i].unsqueeze(0)], [pred_instance.pred_boxes]).to(current_features[0].device)                 # 可能可以直接从 14*14 裁剪   pool from clip fmap    512,768,14,14

                    clip_object_features = clip_object_features.view(
                        clip_object_features.shape[0], clip_object_features.shape[1], -1).permute(0, 2, 1).contiguous()                 # 49,784,768
                    vit_object_features = vit_object_features.view(
                        vit_object_features.shape[0], vit_object_features.shape[1], -1).permute(0, 2, 1).contiguous()                 # 49,784,768
                    
                    global_feature = self.global_mapping(torch.mean(clip_object_features.float(), -2, keepdim=True))     

                    object_features = torch.cat([vit_object_features,clip_object_features],-1)

                    object_features = torch.cat([object_features, global_feature], -2)

                    object_features = self.map_img_to_text(object_features)

                    boxes = pred_instance.pred_boxes.tensor      
                    x = self.visn_fc(object_features)
                    x = self.visn_layer_norm(x)
                    y = self.box_fc(boxes)
                    y_ = self.box_pos_embed.weight.repeat(y.shape[0], 1) 
                    y = y + y_
                    y = self.box_layer_norm(y)

                    object_features = torch.cat([x, y.unsqueeze(1)], 1)


                    object_mask = torch.ones(object_features.size()[:-1],dtype=torch.long).to(object_features.device)                   

                    topk_ids, topk_probs = self.generation(object_features, object_mask)                   # 38 instances in this pic (instance as batch level)   topk_probs: log(confidence)
                    # text_decoder_output = self.tokenizer.decode(topk_ids[0][0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()          # a woman sitting at a desk in an office     [i][j]  [batch_index][output_beam_index]

                    pred_sentences = []
                    logprobs = []
                    for sentence in topk_ids:
                        sentence_ = self.tokenizer.decode(sentence[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").replace(".","").strip()
                        pred_sentences.append(sentence_)
                    
                    for logprob in topk_probs:
                        logprobs.append(logprob[0])

                    assert len(pred_sentences) == len(logprobs)

                    # DenseCap
                    sigmoid = nn.Sigmoid() 
                    cap_scores = logprobs
                    
                    for i,logprob in enumerate(logprobs):
                        cap_scores[i] = torch.exp(logprob)

                    
                    cap_scores = torch.Tensor(cap_scores).to(object_features.device)

                    pred_instance.scores =  (pred_instance.scores * cap_scores) ** 0.5               
                                                        
                    pred_instance.pred_object_descriptions = ObjDescription(pred_sentences)
                                      
                else:
                    pred_instance.pred_object_descriptions = ObjDescription([])

            return pred_instances


    def forward(self, features, proposals, targets=None, targets_task="ObjectDet", clip_features=None):
        if self.training:
            proposals = self.label_and_sample_proposals(
                proposals, targets)

            losses = self._forward_box(features, proposals, targets, targets_task, clip_features)            

            if targets[0].has('gt_masks'):
                mask_losses = self._forward_mask(features, proposals)
                losses.update({k: v * self.mask_weight \
                    for k, v in mask_losses.items()})
            else:
                losses.update(self._get_empty_mask_loss(device=proposals[0].objectness_logits.device))

            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, task=self.test_task, clip_features=clip_features)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    @torch.no_grad()
    def generation(self, question_states, question_atts, out_size=1):
        if self.beam_generator == None:
            self.beam_generator = TextGenerator(self.mplug_config, self.text_decoder)

            print('beam_generator building')
        encoder_inputs = [question_states, question_atts]                                                                
        topk_ids,topk_probs = self.beam_generator.translate_batch_scst(encoder_inputs,out_size=out_size)                 
        return topk_ids, topk_probs
