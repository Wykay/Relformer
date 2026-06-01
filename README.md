# Relformer

> Relation-aware dense captioning built on top of [GRiT](https://github.com/JialianW/GRiT).

<!-- TODO: replace with paper/arXiv/project links once available
**Paper:** [TODO add arXiv link]
**Project page:** [TODO]
-->

Relformer extends GRiT (Generative Regional Transformer, Microsoft 2022) with relation modeling between detected regions, enabling dense captioning that conditions each region's caption on its visual context. The detection backbone follows GRiT (CenterNet2 proposals over a ViT-FPN), the text decoder follows mPLUG, and a CLIP ViT-B/16 branch supplies global visual context.

> **Status.** Research code, work in progress. Trained checkpoints and full reproduction recipes are not yet released. APIs and configs may change.

## Highlights

- **Region-relation ROI head** (`CatAndCat` in `grit/modeling/roi_heads/clip_obj_relation_roi_heads.py`) fuses per-region features with cross-region attention before captioning.
- **CLIP-conditioned meta-arch** (`GRiT_w_CLIP`) injects CLIP ViT-B/16 features alongside the detection backbone.
- **DeepSpeed ZeRO-1 trainer** with a hand-rolled training loop, custom ViT layer-wise LR decay, and multi-node launch helper.
- **Multi-dataset registry**: Visual Genome (`vg`, `vg_2`), VG+COCO joint splits (`vg_coco`), GRiT-COCO, and Object365 are all wired in via `grit/data/datasets/`.

## Repository layout

```
Relformer/
├── train_deepspeed.py        # Training entry point (DeepSpeed)
├── inference.py              # Eval-only entry point
├── launch_deepspeed.py       # Distributed-launch helpers
├── training_args.txt         # Default training args (JSON, see below)
├── inference_args.txt        # Default inference args (JSON)
├── configs/                  # YAML model configs (Detectron2-style)
│   ├── Base.yaml
│   ├── GRiT_B_DenseCap.yaml
│   └── caption_mplug_large.yaml
├── grit/                     # Models, datasets, evaluators, solver
│   ├── modeling/
│   │   ├── meta_arch/        # GRiT, GRiT_w_CLIP + mPLUG/CLIP submodules
│   │   ├── roi_heads/        # CatAndCat & friends, beam search
│   │   ├── backbone/         # ViT, FPN, position encodings
│   │   └── text/             # BERT-based text decoder
│   ├── data/                 # Dataset mappers, augmentations, split registry
│   ├── evaluation/           # COCO / VG dense-caption evaluators
│   ├── config.py             # add_grit_config()
│   └── custom_solver.py      # ViT layer-wise LR decay optimizer
└── third_party/
    └── CenterNet2/           # Vendored proposal generator
```

## Installation

The code targets **Python 3.8 + PyTorch ≥ 1.10 + CUDA**. Versions in `requirements.txt` are pinned to match the upstream GRiT release; bumping them (especially `transformers`, `deepspeed`, `timm`) is likely to break things.

```bash
# 1. Create env
conda create -n relformer python=3.8 -y
conda activate relformer

# 2. Install PyTorch matching your CUDA, e.g.:
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 3. Install Detectron2 (must be done after torch; see Detectron2 docs)
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# 4. Project deps
pip install -r requirements.txt

# 5. Vendored CenterNet2 is added to sys.path at runtime — no install needed.
```

You also need:

- A local copy of `bert-base-uncased`. The default meta-arch loads it from `~/.cache/huggingface/bert-base-uncased` with `local_files_only=True`. Either pre-download it there, or change the path in `grit/modeling/meta_arch/grit.py`.
- An MAE-pretrained ViT-B checkpoint, e.g. `detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth` (referenced from `configs/GRiT_B_DenseCap.yaml`).

## Data layout

Datasets are registered at import time in `grit/data/datasets/`. They expect this layout (symlink `datasets/` into the repo root):

```
datasets/
├── vg/
│   ├── images/
│   └── annotations/
│       └── v1.2/
│           ├── train.json
│           └── test.json
└── coco/                     # only if using grit_coco / vg_coco splits
    ├── train2017/
    ├── val2017/
    └── annotations/
```

Available split names (use them in `DATASETS.TRAIN` / `DATASETS.TEST`):

| Split key       | Defined in                          |
|-----------------|-------------------------------------|
| `vg_train` / `vg_test`         | `grit/data/datasets/vg.py`       |
| `vg_2_train` / `vg_2_test`     | `grit/data/datasets/vg2.py`      |
| `vg_coco_train` / `vg_coco_test` | `grit/data/datasets/vg_coco.py` |
| `grit_coco_*`   | `grit/data/datasets/grit_coco.py`   |
| Object365       | `grit/data/datasets/object365.py`   |

## Usage

Both entry points read a JSON args file (legacy GRiT convention) instead of taking all options on the CLI. Edit `training_args.txt` / `inference_args.txt` in the repo root, or pass `--args-file path/to/your.json`.

### Training

```bash
python train_deepspeed.py
# or with a custom args file
python train_deepspeed.py --args-file my_train_args.json
```

`training_args.txt` controls config selection and run mode:

```json
{
  "config_file": "configs/GRiT_B_DenseCap.yaml",
  "resume": true,
  "eval_only": false,
  "num_gpus": 1,
  "num_machines": 1,
  "machine_rank": 0,
  "num_gpus_per_machine": 1,
  "opts": [],
  "output_dir_name": "./output/debug"
}
```

The actual per-GPU batch size comes from `DATALOADER.DATASET_BS` in the YAML (not `IMS_PER_BATCH`). Effective batch size = `num_machines * num_gpus_per_machine * DATASET_BS`. DeepSpeed is initialized inline in `train_deepspeed.py::get_deep_speed_config` (ZeRO stage 1, fp16 off by default) — edit that function rather than a separate JSON.

### Evaluation

```bash
python inference.py
# or
python inference.py --args-file my_inference_args.json
```

`inference_args.txt`:

```json
{
  "config_file": "configs/GRiT_B_DenseCap.yaml",
  "resume": true,
  "eval_only": true,
  "num_gpus": 1,
  "num_machines": 1,
  "opts": ["MODEL.WEIGHTS", "output/your_run/model_final.pth"],
  "output_dir_name": "./output/inference/debug",
  "test_task": ""
}
```

The evaluator is dispatched from the dataset's `evaluator_type`: `coco` → `GRiTCOCOEvaluator`, `vg` → `GRiTVGEvaluator` (`grit/evaluation/eval.py`).

### Multi-node training

Set `num_machines > 1` in `training_args.txt`. The launcher resolves the master IP from `AZ_BATCH_HOST_LIST`, `AZ_BATCHAI_JOB_MASTER_NODE_IP`, or `MASTER_IP` (in that order); set whichever fits your cluster.

## Architecture overview

```
                ┌──────────────────────────────────────┐
                │  Meta-arch (MODEL.META_ARCHITECTURE) │
                │  GRiT_w_CLIP  /  GRiT                │
                └──────────────┬───────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
  ┌─────────────────────┐         ┌────────────────────────┐
  │ ViT-FPN backbone    │         │ CLIP ViT-B/16 (frozen) │
  │ (build_vit_fpn_     │         │ — global context only  │
  │  backbone)          │         └────────────┬───────────┘
  └──────────┬──────────┘                      │
             │                                 │
             ▼                                 │
  ┌─────────────────────┐                      │
  │ CenterNet2 proposal │                      │
  │ generator           │                      │
  └──────────┬──────────┘                      │
             │                                 │
             ▼                                 │
  ┌──────────────────────────────────────────┐ │
  │ ROI heads (MODEL.ROI_HEADS.NAME)         │◄┘
  │   CatAndCat → cross-region attention →   │
  │   mPLUG text decoder → beam search       │
  └──────────────────────────────────────────┘
```

Wiring details worth knowing before extending:

- **Meta-arch is YAML-selected.** `MODEL.META_ARCHITECTURE` (default `GRiT_w_CLIP`) decides whether `clip_meta_arch.py::GRiT_w_CLIP` or `grit.py::GRiT` is built. Both load `configs/caption_mplug_large.yaml` and graft `MPLUG.text_decoder` onto `roi_heads.text_decoder` — any new ROI head must accept that attribute.
- **Config plumbing.** `setup()` calls `add_centernet_config(cfg)` then `add_grit_config(cfg)` before `cfg.merge_from_file`. New config keys must be declared in `grit/config.py::add_grit_config` to survive `cfg.freeze()`.
- **Custom solver.** `SOLVER.USE_CUSTOM_SOLVER=True` (default) routes optimizer construction through `grit/custom_solver.py`, which applies ViT layer-wise LR decay (`SOLVER.VIT_LAYER_DECAY`, `SOLVER.VIT_LAYER_DECAY_RATE`).
- **Registry imports.** Adding a new ROI head / meta-arch / backbone requires importing it from `grit/__init__.py`; otherwise its `@*_REGISTRY.register()` decorator never fires and Detectron2 cannot resolve the YAML name.

## Known issues

- The training/inference loops use a hand-rolled `do_train` instead of Detectron2's `DefaultTrainer`. Hooks like EMA and gradient accumulation are not wired up.
- Some ROI head modules referenced by older checkpoints (`object_relation_roi_heads`, `mplug_roi_heads`, `whole2local_roi_heads`, etc.) survive only as `.pyc` in `__pycache__`; the corresponding `.py` sources are not in this tree. The active head is `CatAndCat` from `clip_obj_relation_roi_heads.py`.
- BERT tokenizer paths are hard-coded; see Installation.

## Acknowledgements

Built on top of:

- [GRiT](https://github.com/JialianW/GRiT) — base detection + dense-caption framework (Wu et al., 2022)
- [Detectron2](https://github.com/facebookresearch/detectron2) — detection toolbox
- [CenterNet2](https://github.com/xingyizhou/CenterNet2) — proposal generator (vendored under `third_party/`)
- [mPLUG](https://github.com/alibaba/AliceMind/tree/main/mPLUG) — text decoder
- [CLIP](https://github.com/openai/CLIP) — visual context encoder

## License

This project is released under the MIT License (see [LICENSE](LICENSE)). It inherits the GRiT MIT license from Microsoft Corporation, 2022.

## Citation

<!-- TODO: fill in once the paper is on arXiv. -->

```bibtex
@misc{relformer,
  title  = {Relformer: Relation-aware Dense Captioning},
  author = {TODO},
  year   = {TODO},
  note   = {TODO arXiv link}
}
```

If you use this codebase, please also cite the upstream GRiT work:

```bibtex
@article{wu2022grit,
  title  = {GRiT: A Generative Region-to-text Transformer for Object Understanding},
  author = {Wu, Jialian and Wang, Jianfeng and Yang, Zhengyuan and Gan, Zhe and Liu, Zicheng and Yuan, Junsong and Wang, Lijuan},
  journal = {arXiv preprint arXiv:2212.00280},
  year   = {2022}
}
```
