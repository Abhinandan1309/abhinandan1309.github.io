---
layout: post
title: "INT8 breaks popular edge models. Here's why, and an exact fix"
description: "EfficientNet, MobileNetV3, LCNet and MobileViT lose most of their accuracy under default INT8 on Qualcomm, TI, AMD and NVIDIA toolchains. Why it happens, and a rewrite that fixes it without retraining."
---

In my master's thesis at KIT, I deployed LiDAR segmentation networks on a TI TDA4VM, a chip
built for cars. I got them running, but only with workarounds: 16-bit precision, and swapping
out a layer the toolchain mishandled. The fastest mode, 8-bit, never worked. One model fell from
83% to 35% accuracy, even with the vendor's advanced calibration. I wrote "mixed precision" under
future work and moved on.

That unsolved problem stayed with me. So I went back to INT8 on edge chips, this time properly:
four vendors, eight models, every result measured.

![TIDL 8-bit segmentation masks break; with Anneal they nearly match FP32](/assets/img/segmentation.png)

*LRASPP-MobileNetV3 on TI's TDA4VM (TIDL emulation), 300 COCO images: FP32 54.4 mIoU,
TIDL 8-bit 9.2, TIDL 8-bit with Anneal 53.2.*

## What I found

I ran 8 image classifiers through the default INT8 path of four vendors: Qualcomm (Galaxy S24
NPU), TI (TDA4VM), AMD (XINT8, the Ryzen AI NPU's arithmetic) and NVIDIA (TensorRT with
ModelOpt).

INT8 should cost about a point of accuracy. For EfficientNet, MobileNetV3, LCNet and MobileViT
it commonly cost 35 to 77 points, which means the model stops working. ResNet-50, the control,
lost at most 4.6.

## Why it happens

The answer turned out to be almost embarrassingly simple.

These networks feed a gate (SiLU or Hardswish) into a depthwise convolution. The gate's output
channels have very different ranges: up to 360× apart in EfficientNet-B0. INT8 gives the whole
tensor one scale, set by the largest channel. The smallest channels get less than one INT8
level, so they round to zero.

![Per-channel INT8 levels in EfficientNet-B0, before and after](/assets/img/channel_ranges.png)

## The fix

The fix is a rewrite of the model that changes nothing in float. Scale each channel by a factor
s before the gate, let the gate read the scaled value divided by s (so it sees exactly what it
saw before), and divide s back out in the next convolution. The model computes the same output
as before, so there is nothing to retrain. But now every channel has a similar range and keeps
its INT8 resolution.

![The rewrite](/assets/img/method.png)

It goes in before the vendor's toolchain, which I use unchanged:

![Where Anneal sits](/assets/img/workflow.png)

Each toolchain needed a little more. On TI, the first few layers run in 16 bits, using TIDL's
own option. That is the mixed precision from my thesis's future-work list, a few years late.
On AMD, the gates run in 16 bits, with bias correction. On the S24 and T4, Anneal can also
produce the INT8 model itself and hand it to the vendor's compiler.

## Did it work?

![8 models × 4 toolchains, before and after](/assets/img/grid.png)

Top-1 change vs FP32, percentage points, 300–1,500 Imagenette validation images per cell,
paired against FP32 on the same images. The S24 and T4 are real devices. The AMD and TI
numbers come from the vendors' own quantizers and emulators on a PC.

- 26 of 32 model–toolchain pairs end within 2.2 points of FP32 (24 within 2.0).
- EfficientNet-B0: −75 → −0.1 (AMD), −12.8 → −0.7 (S24), −73 → −1.6 (TI), −52 → −0.1 (T4).
- MobileNetV3-Small: −58 to −66 → −1.4 to −2.1 on all four.

## What it costs

![Galaxy S24 latency and accuracy](/assets/img/speed_s24.png)

On the S24's NPU, the rewritten INT8 model is slower than plain INT8 but still faster than FP16,
and far more accurate than plain INT8. On an NVIDIA T4 at batch 1, every INT8 engine I built,
the vendor's or mine, was slower than plain FP16 for these small models. There, FP16 is simply
the better choice.

On TI's chip I don't have a board, so I used TI's performance simulator on the segmentation
model. Plain 8-bit: 12.0 ms per frame. With the fix: 13.8 ms, about 14% slower. Full 16-bit:
21.5 ms, 78% slower. Most of the extra time comes from the few early layers kept at 16-bit.
So the fix gets close to 16-bit accuracy for about a fifth of 16-bit's extra time.

## What I couldn't fix

- EfficientNet-B1 on TI: still −8.9 points. TI's full 16-bit mode loses only 0.9, at 16-bit cost.
- MobileViT on TI: −4.2. TI's own 16-bit mode does better (−2.6).
- EfficientViT-B0 on TensorRT: −70 → −12.8. Better, not good.
- Intel OpenVINO doesn't have the problem at all, so the rewrite doesn't help there.

## Try it

The tool is open source: [Anneal on GitHub](https://github.com/Abhinandan1309/anneal).

```python
from pathlib import Path
from anneal.core.equalize import equalise

# batches: a few preprocessed calibration batches, float32 NCHW numpy arrays
equalise(Path("model.onnx"), Path("model-eq.onnx"), batches,
         residual=True, se=True, grid_inverse=True)
```

Every number here, including the failures, comes from a result file in the repository, with the
script that produced it. The [benchmark grid](https://github.com/Abhinandan1309/anneal/blob/main/docs/benchmark_grid.md)
lists the recipe behind each cell.

If you deploy models to edge hardware and have hit this, or something like it, I'd like to hear
about it.

---

*Segmentation images: COCO val2017 photos from Flickr under CC BY 2.0, resized and overlaid with
masks. Credits: [list](https://github.com/Abhinandan1309/anneal/blob/main/docs/figures/segmentation_credits.txt).*
