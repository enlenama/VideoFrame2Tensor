# VideoFrame2Tensor

## Overview

VideoFrame2Tensor provides an efficient pipeline for importing video frames directly into machine learning tensors, eliminating CPU-GPU data copies and  big buffer transfers. It leverages hardware-accelerated preprocessing using D3D12 APIs and DirectML operators for real-time ML applications on Windows.

## Directory Structure

- `explainer.md` — Technical explainer and API usage details
- `example/semantic_segmentation/` — Example implementation for real-time video semantic segmentation
- `test-data/` — Contains test models and data for evaluation
