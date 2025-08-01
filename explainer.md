# Import Video Frame to Tensor

## Authors

* [Yingying Ma](yingying.ma@intel.com)

## Participate

* Issue track TBA

## Introduction

This explainer proposes a `WriteTensor` interface overload that can directly import a video frame to an `MLTensor`. This enhancement provides more efficient pipeline for importing [VideoFrame](https://www.w3.org/TR/webcodecs/#videoframe-interface) to machine learning tensors, and eliminating the overhead of readbacking from accelerator memory, and transferring of big buffers across processes.

Currently, processing video frames for ML inference requires:

- Copying back video frame from GPU/NPU to CPU.
- Pixel data pre-processing(such as scaling, cropping, pixel format conversion and normalization) in JavaScript layer.
- Transferring big buffers from Javascript layer to WebNN inference engine.
- Re-uploading processed data to GPU for inference.

The proposed solution simplies these steps by:

- Transferring shared accelerator resource handles rather than big buffers in browsers.
- Using GPU for hardware accelerated pre-processing required for importing from VideoFrame to MLTensor.

## Goals

- **Performance**: Eliminate unnecessary readback from accelerator memory when processing video frames for ML inference.
- **Hardware Acceleration**: Leverage hardware accelerated fixed-function pre-processing with lower power consumption, and leave more GPU/NPU resource to AI inference workloads.
- **Compatibility**: Maintain compatibility with existing WebNN APIs.

## Non-goals

- Replacing existing tensor creation methods.
- Gurantee zero-copy buffer sharing between VideoFrame and WebNN.
- Provides capability of exporting MLTensor to VideoFrame.

## Other Solutions

In [MLTensor explainer](https://github.com/webmachinelearning/webnn/blob/main/mltensor-explainer.md) the interface for interop between WebGPU and WebNN is proposed for buffer sharing between these two APIs.

While WebGPU offers the flexibility to customize the pre-processing required for MLTensor import, these pre-processing will be power consuming comparing to fixed-function video-processing hardware.

## Using the API

### Interface Definition

```webidl
dictionary MLVideoFrame {
  required VideoFrame frame;
  required boolean isNchw;
  required DOMRect visibleRect;
  required float mean;
  required float std;
};

partial interface MLContext {
  [
    RuntimeEnabled=MachineLearningNeuralNetwork,
    CallWith=ScriptState,
    RaisesException
  ] void writeTensor(MLTensor dstTensor, MLVideoFrame srcFrame);
};
```

**Parameter Details:**

- **`frame`**: The source video frame.
- **`isNchw`**: Indicates the layout of the destination `MLTensor`. Set to `true` for NCHW format, or `false` for NHWC format.
- **`visibleRect`**: Defines the destination visible rectangle region of the video frame.
- **`mean`**, **`std`**: Used for pixel normalization. Each pixel value is normalized as `(pixel - mean) / std`, with `mean` and `std` applied uniformly across all RGB channels (e.g., 127.5 for [0,255] → [-1,1]).

### Basic Usage

Below is a step-by-step example showing how to efficiently import a video frame captured by camera into a tensor for real-time semantic segmentation:

- **Request camera stream**: Use `getUserMedia()` to access the device's camera and obtain a video stream.

  ```javascript
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  const video = document.createElement('video');
  video.srcObject = stream;
  ```
- **Create a VideoFrame**: Utilize WebCodecs API `VideoFrame`  to wrap the current frame from the video element, allowing direct GPU access and efficient processing.

  ```javascript
  const videoFrame = new VideoFrame(video);
  ```
- **Initialize WebNN context and tensor**: The WebNN context is created with GPU as the device type to ensure hardware acceleration. The tensor is allocated with the required shape and usage for ML inference.

  ```javascript
  const context = await navigator.ml.createContext({ deviceType: 'gpu' });
  const tensor = await context.createTensor({
    dataType: 'float32',
    dimensions: [1, height, width, 3], // NHWC format
    usage: MLTensorUsage.WRITE
  });
  ```
- **Prepare MLVideoFrame parameters**: Set up the parameters for preprocessing, including layout, visible region, and normalization values.

  ```javascript
  const mlVideoFrame = {
    frame: videoFrame,
    isNchw: false, // Layout of the destination ml tensor is NHWC.
    visibleRect: new DOMRect(0, 0, video.videoWidth, video.videoHeight),
    mean: 127.5,
    std: 127.5
  };
  ```
- **Import video frame to tensor**: The `writeTensor` method transfers the video frame handle directly to the tensor, eliminating CPU-GPU data copies and big buffer transfer.

  ```javascript
  context.writeTensor(tensor, mlVideoFrame);
  ```
- **Inference**: Dispatch the ML graph using the imported tensor as input. The output contains the inference results, such as segmentation masks.

  ```javascript
  const outputs = await context.dispatch(graph, { input: tensor }, { output: outputTensor });
  ```

This approach ensures minimal latency and optimal performance for real-time video ML applications.

## Key Scenarios

### Real-time Video Semantic Segmentation

This scenario focuses on processing live camera streams for real-time semantic segmentation applications. The typical workflow involves:

- **Camera Stream Capture**: Capture a video frame from live camera stream.
- **Hardware-accelerated video processing**: Use accelerator APIs provided by OS to efficiently perform input preprocessing tasks, including frame size scaling, pixel format conversion, cropping and normalization, etc.
- **GPU-Accelerated Inference**: Leveraging WebNN with GPU backend for fast inference.
- **Visual Feedback**: Rendering with segmentation masks utilized on the original video stream, for example, perform blur or replace backgrounds with static/dynamic images.

**Key Benefits**:

**Zero-Copy Processing**: For the destination tensor with NHWC layout, all frame preprocessing is performed entirely on the GPU using D3D12 APIs.

**One-Copy Processing**: For the destination tensor with NCHW layout, most preprocessing is performed on the GPU using D3D12 APIs, with only a single readback to the CPU required for layout conversion.

> Alternatively, a web application can choose to initialize the ML tensor in NHWC layout and insert a transpose layer at the beginning of the ML graph to convert the input to the required model layout (e.g., NCHW). This approach allows all preprocessing to remain on the GPU and avoids CPU readback for layout conversion.

**Low Latency**: Eliminates CPU-GPU data copy and inter-process big buffer transfers each frame.

**Use Cases**:

- Real-time background replacement in video conferencing
- Augmented reality applications with scene understanding
- Privacy protection through selective content masking

## Examples

For a complete working implementation of real-time video semantic segmentation using the VideoFrame2Tensor API, please refer to the example in the `example/semantic_segmentation` directory.

## Implementation Details

### Windows

#### D3D12 Video Processor on Win

- Provides hardware-accelerated video frame processing on Windows platforms. Common operations include:
  - Frame size scaling: Efficiently resize video frames to match model input dimensions.
  - Pixel format conversion: Convert between various color formats (e.g., NV12, RGBA) directly on the GPU.
  - Cropping: Select visible regions for processing without extra memory copy.
  - Buffer sharing: Avoids unnecessary CPU-GPU transfers by keeping frames in GPU memory throughout preprocessing.
  - Seamless integration with DirectML for further ML-specific operations.

#### DirectML Operator on Win

- Enables efficient ML-specific preprocessing on the GPU, including:
  - Data type casting: Convert pixel data to the required tensor type (e.g., uint8 → float32) for inference.
  - Normalization: Apply mean and std normalization to pixel values as required by ML models.

### macOS

#### CoreML Integration

TBA

## Compatibility

- Camera access permissions
- Requires WebNN API support
- Inferencing framework's interop with rendering API used by VideoFrame's backend.
- VideoFrame API compatibility
