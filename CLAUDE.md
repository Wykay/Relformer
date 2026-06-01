# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Relformer is a research codebase extending **GRiT** (Generative Regional Transformer, Microsoft 2022) with relation modeling between regions for dense image captioning. It is built on Detectron2 with DeepSpeed (ZeRO-1) for training, an mPLUG text decoder for caption generation, and a CLIP ViT-B/16 branch for global visual context. The default branch is `v1`. See `README.md` for the user-facing overview.

## Common commands

There is no Makefile, no test suite, and no linter configuration. The two entry points are:

```bash
# Training (DeepSpeed; also handles eval-only when configured)
python train_deepspeed.py [--args-file path/to/args.json]

# Single-GPU inference / evaluation
python inference.py [--args-file path/to/args.json]
```

Both entry points still load most of their configuration from a JSON args file (legacy GRiT convention). `--args-file` defaults to `training_args.txt` / `inference_args.txt` in the repo root. Edit those (or pass a custom one) to change `config_file`, `eval_only`, `num_gpus`, `num_machines`, `output_dir_name`, `opts`. CLI flags other than `--args-file` are clobbered by the JSON.

To run a different model config, point the JSON's `config_file` at e.g. `configs/GRiT_B_DenseCap.yaml` (which inherits from `configs/Base.yaml`).

## Required runtime layout

The code assumes several paths exist outside the repo. New instances will hit hard failures otherwise:

- `datasets/` (symlinked into the repo root) holding e.g. `datasets/vg/images` and `datasets/vg/annotations/v1.2/{train,test}.json` — see `_CUSTOM_SPLITS_LVIS` at the bottom of each `grit/data/datasets/vg*.py` for the exact expected layout per split (`vg`, `vg_2`, `vg_coco`, `grit_coco`, `object365`).
- `~/.cache/huggingface/bert-base-uncased` — loaded by `grit/modeling/meta_arch/grit.py` via `BertTokenizer.from_pretrained(..., local_files_only=True)`. `clip_meta_arch.py` instead pulls `google-bert/bert-base-cased` from the Hub.
- A pretrained ViT checkpoint, e.g. `detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth` (set in `MODEL.WEIGHTS`).

## Pinned dependencies

`requirements.txt` pins versions that the rest of the stack assumes — do not bump casually. Notably: `transformers==4.21.1`, `deepspeed==0.7.0`, `timm==0.6.7`, `einops==0.4.1`, `Pillow==9.0.0`, `pydantic==1.9.1`, `protobuf==3.19.6`. Detectron2 itself is **not** in `requirements.txt` and must be installed separately (see README). CenterNet2 is vendored under `third_party/CenterNet2/` and added to `sys.path` at runtime by both entry scripts:

```python
sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
```

Always run training/inference from the repo root so this relative path resolves.

## Architecture

The model is a Detectron2 `GeneralizedRCNN` extended with a generative text decoder. It is assembled through Detectron2's registry pattern, so new components are wired in by importing them from `grit/__init__.py` (which is what makes `*_REGISTRY.register()` decorators take effect).

```
            +-------------------- Meta-arch (META_ARCHITECTURE) --------------------+
            |  GRiT_w_CLIP (default in Base.yaml)  |  GRiT (used by DenseCap cfg)    |
            |  - clip_backbone (ViT-B/16)          |  - mPLUG text_decoder           |
            |  - mPLUG text_decoder                |                                 |
            +----------------------------+---------+---------------------------------+
                                         |
            +----------------------------v----------------------------+
            |   Backbone: build_vit_fpn_backbone (grit/modeling/      |
            |   backbone/vit.py, position_encoding.py, utils.py)      |
            +----------------------------+----------------------------+
                                         |
            +----------------------------v----------------------------+
            |   Proposal generator: CenterNet (third_party/CenterNet2)|
            +----------------------------+----------------------------+
                                         |
            +----------------------------v----------------------------+
            |   ROI heads (NAME=CatAndCat by default)                 |
            |   grit/modeling/roi_heads/                              |
            |     grit_roi_heads.py (GRiTROIHeadsAndTextDecoder)      |
            |     clip_obj_relation_roi_heads.py (CatAndCat)          |
            |   The chosen head receives the mPLUG text_decoder       |
            |   from the meta-arch and runs caption beam search       |
            |   (grit_fast_rcnn.py + beam_search.py).                 |
            +---------------------------------------------------------+
```

Key wiring details that span files:

- **Config plumbing.** `setup()` in both entry scripts calls `add_centernet_config(cfg)` then `add_grit_config(cfg)` (`grit/config.py`) before merging the YAML. New config keys must be declared in `add_grit_config` to survive `cfg.freeze()`.
- **Meta-arch is YAML-selected.** `MODEL.META_ARCHITECTURE` (default `GRiT_w_CLIP` from `Base.yaml`) determines whether `grit/modeling/meta_arch/clip_meta_arch.py` or `grit.py` is instantiated. Both internally load `configs/caption_mplug_large.yaml` and attach an mPLUG decoder onto `self.roi_heads.text_decoder` — so a new ROI head must expose that attribute.
- **mPLUG / text decoder.** Lives under `grit/modeling/meta_arch/mplug/` and `grit/modeling/roi_heads/mplug/`. The `MPLUG` class in `model_caption_mplug.py` is constructed only to graft its `text_decoder` onto the ROI heads; the rest of MPLUG is unused at runtime.
- **Datasets.** Registered at import time. `grit/__init__.py` imports `grit.data.datasets.{vg, vg2, vg_coco, grit_coco, object365}`, each of which calls `DatasetCatalog.register(...)` for splits like `vg_train`, `vg_2_train`, `vg_coco_train`. To add a split, follow the pattern in `vg2.py` (`_CUSTOM_SPLITS_LVIS` + `register_vg_instances`).
- **Evaluation dispatch.** `do_test` in `train_deepspeed.py` / `inference.py` picks `GRiTCOCOEvaluator` or `GRiTVGEvaluator` (`grit/evaluation/eval.py`) based on the registered `evaluator_type` of the dataset. Anything else raises `NotImplementedError`.
- **Custom optimizer.** `grit/custom_solver.py::build_custom_optimizer` is enabled by `SOLVER.USE_CUSTOM_SOLVER=True` (default). It applies ViT layer-wise LR decay (`SOLVER.VIT_LAYER_DECAY`, `SOLVER.VIT_LAYER_DECAY_RATE`).

## Training-loop specifics

`do_train` in `train_deepspeed.py` is hand-rolled rather than using Detectron2's `DefaultTrainer`. Important behaviors:

- `deepspeed.initialize` is called with an inline config from `get_deep_speed_config` — ZeRO stage 1, fp16 disabled, no flops profiler. Edit that function (not a separate config file) to change DeepSpeed behavior.
- `train_batch_size = num_gpus_per_machine * cfg.DATALOADER.DATASET_BS` (single-node) — `IMS_PER_BATCH` in YAML is **not** the source of truth for batch size.
- Periodic eval is gated by `cfg.TEST.EVAL_PERIOD`; checkpoints by `cfg.SOLVER.CHECKPOINT_PERIOD` (default 10000).
- Multi-node training uses MPI-style env vars and resolves the master IP via `AZ_BATCH_HOST_LIST` / `AZ_BATCHAI_JOB_MASTER_NODE_IP` / `MASTER_IP`. It is launched through `launch_deepspeed.py::launch_deepspeed_multinodes`.

## Gotchas

- `cfg.freeze()` happens early in `setup()`; new config keys must be added via `add_grit_config` or merging will fail.
- The original GRiT release left around several ROI-head modules (`object_relation_roi_heads`, `clip_roi_heads`, `mplug_roi_heads`, `whole2local_roi_heads`, `obj_rela_roi_heads_with_cross`, `rela_debug`, `relformer_clip_encode_region`) whose `.py` sources are not in this tree. `grit/__init__.py` no longer imports them — only the active heads (`grit_roi_heads`, `clip_obj_relation_roi_heads`) are loaded. Same for `backbone.ria`. If a checkpoint references one of those classes, you'll need to restore the source before loading.
- The launcher file used to be misspelled `lauch_deepspeed.py`; it is now `launch_deepspeed.py`. Old branches/checkouts may still use the typo'd name.
- Entry scripts used to hard-code `/opt/data/private/jwq/model_densecap/...` for the args JSON; that path is gone, replaced by `--args-file` defaulting to repo-root `training_args.txt` / `inference_args.txt`.
