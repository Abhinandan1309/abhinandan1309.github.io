---
layout: post
title: "INT8 breaks popular edge models. An exact fix, tested on four toolchains"
description: "EfficientNet, MobileNetV3, LCNet and MobileViT lose most of their accuracy under default INT8 on Qualcomm, TI, AMD and NVIDIA toolchains. Why, and a rewrite that fixes it without retraining."
---

Quantizing a model to INT8 should cost a point of accuracy, not most of it. For a family of
popular edge models it costs most of it, on almost every toolchain I tried.

![TIDL 8-bit segmentation masks break; with Anneal they nearly match FP32](/assets/img/segmentation.png)

*LRASPP-MobileNetV3 on TI's TDA4VM (TIDL emulation), 300 COCO images: FP32 54.4 mIoU,
TIDL 8-bit 9.2, TIDL 8-bit with Anneal 53.2.*

## The problem

I tested 8 image classifiers on the default INT8 path of four vendors: Qualcomm (Galaxy S24
NPU), TI (TDA4VM), AMD (XINT8, the Ryzen AI NPU's arithmetic) and NVIDIA (TensorRT with
ModelOpt). EfficientNet, MobileNetV3, LCNet and MobileViT commonly lost 35 to 77 points of top-1
accuracy. ResNet-50, the control, lost at most 4.6.

## Why

These networks feed a gate (SiLU or Hardswish) into a depthwise convolution. The gate's output
channels have very different ranges: up to 360× apart in EfficientNet-B0. INT8 gives the whole
tensor one scale, so it is set by the largest channel, and the smallest ones get less than one
INT8 level. They round to zero.

![Per-channel INT8 levels in EfficientNet-B0, before and after](/assets/img/channel_ranges.png)

## The fix

Scale each channel by a factor s before the gate, let the gate read the scaled value divided by
s (so it sees exactly what it saw before), and divide s back out in the next convolution. The
float model computes the same output as before. No retraining, no data beyond a few calibration
images. After the rewrite, every channel has a similar range and keeps its INT8 resolution.

![The rewrite](/assets/img/method.png)

The rewrite goes in before the vendor's toolchain, which is used unchanged:

![Where Anneal sits](/assets/img/workflow.png)

Each toolchain needs a little more than the rewrite. On TI, the first few layers run in 16 bits
(TIDL's own option). On AMD, the gates run in 16 bits, with bias correction. On the S24 and T4,
Anneal can also produce the INT8 model itself and hand it to the vendor's compiler.

## Results

![8 models × 4 toolchains, before and after](/assets/img/grid.png)

Top-1 change vs FP32, percentage points, 300–1,500 Imagenette validation images per cell,
paired against FP32 on the same images. The S24 and T4 are real devices. The AMD and TI
numbers come from the vendors' own quantizers and emulators on a PC.

- 26 of 32 model–toolchain pairs end within 2.2 points of FP32 (24 within 2.0).
- EfficientNet-B0: −75 → −0.1 (AMD), −12.8 → −0.7 (S24), −73 → −1.6 (TI), −52 → −0.1 (T4).
- MobileNetV3-Small: −58 to −66 → −1.4 to −2.1 on all four.

## What it costs

![Galaxy S24 latency and accuracy](/assets/img/speed_s24.png)

On the S24 NPU the rewritten INT8 model is slower than plain INT8 but faster than FP16, and far
more accurate than plain INT8. On an NVIDIA T4 at batch 1, every INT8 engine I built, the
vendor's or mine, was slower than plain FP16 for these small models. There, FP16 is the better
choice.

## What is not solved

- EfficientNet-B1 on TI: −8.9 points. TI's full 16-bit mode loses only 0.9, at 16-bit cost.
- MobileViT on TI: −4.2. TI's 16-bit mode does better (−2.6).
- EfficientViT-B0 on TensorRT: −70 → −12.8.
- Intel OpenVINO does not have the problem, and the rewrite does not help there.

## Try it

The tool is open source: [Anneal on GitHub](https://github.com/Abhinandan1309/anneal).

```python
from pathlib import Path
from anneal.core.equalize import equalise

# batches: a few preprocessed calibration batches, float32 NCHW numpy arrays
equalise(Path("model.onnx"), Path("model-eq.onnx"), batches,
         residual=True, se=True, grid_inverse=True)
```

Every number here, including the failures, comes from a result file in the repository, with its
script. The [benchmark grid](https://github.com/Abhinandan1309/anneal/blob/main/docs/benchmark_grid.md)
lists the recipe behind each cell.

---

*Segmentation images: COCO val2017 photos from Flickr under CC BY 2.0, resized and overlaid with
masks. Credits: [list](https://github.com/Abhinandan1309/anneal/blob/main/docs/figures/segmentation_credits.txt).*
