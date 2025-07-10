# EVA Optimization Summary

This issue documents three key bottlenecks identified in EVA’s offline BACK pipeline and proposes corresponding optimization strategies.

---

## 1. Data Loading Bottleneck

**Problem**:
- High overhead due to per-sample `.pt` file disk reads.
- GPU frequently idles while waiting for CPU-side loading.

**Proposed Optimization**:
- Preload embeddings and cache in RAM.

**Effort**: Low 
**Impact**: High 
**Status**: Implemented

---

## 2. Metrics Computation Overhead

**Problem**:
- All metrics (Accuracy, F1, Precision, Recall, AUROC, Loss) computed every training step.
- Slows down training throughput.

**Proposed Optimization**:
- Defer full metric computation to validation phase.
- Track only loss or accuracy during training at lower frequency (e.g. every N steps).

**Effort**: Low 
**Impact**: Medium
**Status**: Future Work

---

## 3. Low GPU Utilization

**Problem**:
- Short GPU bursts followed by long CPU-prep or sync idle periods.
- Model size is small, so overhead dominates per step.

**Proposed Optimization**:
- Fuse small ops (e.g. data preprocessing, augmentation) into single batched GPU operations.
- Use `torch.compile()` to reduce Python overhead and fuse kernels.

**Effort**: High 
**Impact**: Medium 
**Status**: Future work