# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Jialian Wu from https://github.com/facebookresearch/Detic/blob/main/train_net.py
import logging
import os
import sys
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime
import json

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
    LVISEvaluator,
    COCOEvaluator,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from grit.config import add_grit_config
from grit.data.custom_build_augmentation import build_custom_augmentation
from grit.data.custom_dataset_dataloader import  build_custom_train_loader
from grit.data.custom_dataset_mapper import CustomDatasetMapper
from grit.custom_solver import build_custom_optimizer
from grit.evaluation.eval import GRiTCOCOEvaluator, GRiTVGEvaluator


logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    results = OrderedDict()
    for d, dataset_name in enumerate(cfg.DATASETS.TEST):
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' \
            else DatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, False))
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)                 # dataset_name: 'vg_test'
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == 'coco':
            evaluator = GRiTCOCOEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'vg':
            evaluator = GRiTVGEvaluator(dataset_name, cfg, True, output_folder)
        else:
            raise NotImplementedError('We have not implemented the evaluator for {}'.format(evaluator_type))
            
        results[dataset_name] = inference_on_dataset(
            model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.output_dir_name:
        cfg.OUTPUT_DIR = args.output_dir_name
    logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    if args.test_task:
        cfg.MODEL.TEST_TASK = args.test_task

    cfg.freeze()

    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), color=False, name="grit")
    return cfg


def main(args):
    
    cfg = setup(args)
    model = build_model(cfg)                                                                # would move the model to cfg.MODEL.DEVICE automatically
    logger.info("Model:\n{}".format(model))

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        return do_test(cfg, model)

    else:
        print('use train_deepspeed to train model')
        return


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--args-file",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_args.txt"),
        help="JSON file whose contents replace argparse args (legacy GRiT convention).",
    )
    args = parser.parse_args()

    with open(args.args_file, "r") as f:
        args.__dict__ = json.load(f)

    if args.num_machines == 1:
        args.dist_url = 'tcp://127.0.0.1:{}'.format(
            torch.randint(11111, 60000, (1,))[0].item())
    else:
        raise NotImplementedError('Use train_deepspeed.py for multi-node training')
    print("Command Line Args:", args)

    launch(
    main,
    args.num_gpus_per_machine,
    num_machines=args.num_machines,
    machine_rank=args.machine_rank,
    dist_url=args.dist_url,
    args=(args,),
    )




