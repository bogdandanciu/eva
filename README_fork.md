# EVA Fork - Evaluation and Profiling

Fork of [kaiko-ai/eva](https://github.com/kaiko-ai/eva) featuring optimizations for offline classification performance and support for evaluation and profiling.

## Local Setup

```bash
git clone https://github.com/bogdandanciu/eva.git
cd eva
python3 -m venv eva-fork-env
source eva-fork-env/bin/activate
pip install -e '.[all]'
```

## Run Evaluation

Baseline evaluation

```bash
N_RUNS=1 DOWNLOAD_DATA=true MODEL_NAME=universal/vit_small_patch16_224_dino \
eva predict_fit --config https://raw.githubusercontent.com/kaiko-ai/eva/main/configs/vision/pathology/offline/classification/bach.yaml
```

Optimized evaluation (only change is the config file link which has one additional option for preloading)

```bash
N_RUNS=1 DOWNLOAD_DATA=true MODEL_NAME=universal/vit_small_patch16_224_dino \
eva predict_fit --config https://raw.githubusercontent.com/bogdandanciu/eva/main/configs/vision/pathology/offline/classification/bach.yaml
```

## Profiling (Optional)

### Enable PyTorch Profiler

```bash
DOWNLOAD_DATA=false MODEL_NAME=universal/vit_small_patch16_224_dino MAX_STEPS=50 N_RUNS=1 eva predict_fit --config https://raw.githubusercontent.com/bogdandanciu/eva/main/configs/vision/pathology/offline/classification/bach.yaml --trainer.profiler.class_path=lightning.pytorch.profilers.PyTorchProfiler --trainer.profiler.init_args.dirpath=./profiler_traces --trainer.profiler.init_args.filename=eva_bach_profile --trainer.profiler.init_args.export_to_chrome=true --trainer.profiler.init_args.record_shapes=true
```

### Nsight Systems

```bash
nsys profile   --output eva_predict_fit_profile   --cudabacktrace=all   --trace=cuda,nvtx,osrt   --sample=cpu   --stop-on-exit=true   --force-overwrite=true   bash -c "N_RUNS=1 DOWNLOAD_DATA=false MODEL_NAME=universal/vit_small_patch16_224_dino eva predict_fit --config https://raw.githubusercontent.com/bogdandanciu/eva/main/configs/vision/pathology/offline/classification/bach.yaml"
```

---

For more documentation, see the [official EVA docs](https://kaiko-ai.github.io/eva).
