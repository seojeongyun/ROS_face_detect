# ROS Face Detect

## Overview

**YOLOv6 기반 얼굴 탐지 모델을 ROS 환경에서 실시간으로 구동하기 위한 프로젝트**입니다.

영상 내 얼굴을 탐지하는 YOLOv6 모델을 학습하고, ROS의 Publisher/Subscriber 구조를 이용해 카메라 영상의 입력부터 얼굴 탐지 결과 출력까지 실시간 처리 Pipeline을 구성했습니다.

또한 Edge Device 환경에서의 실시간 처리를 고려하여 **입력 Frame 제어, ONNX 변환 및 TensorRT 기반 추론**을 위한 코드를 구현했습니다.

---

## Architecture

```text
Camera
  │
  ▼
ROS Image Topic
/camera/color/image_raw
  │
  ▼
FPS Controller
  │
  ▼
/fps_controller/image_raw
  │
  ▼
YOLOv6 Face Detector
  │
  ├─ Resize / Pre-processing
  ├─ Face Detection
  └─ NMS / Post-processing
  │
  ▼
/face_detector/image_result
  │
  ▼
Detection Result
```

---

## YOLOv6-based Face Detection

YOLOv6를 기반으로 영상 내 얼굴의 Bounding Box를 탐지합니다.

Face Detector는 다음 과정을 수행합니다.

```text
Input Image
    │
    ▼
Pre-processing
    │
    ▼
YOLOv6
    │
    ▼
Face Bounding Boxes
    │
    ▼
NMS & Post-processing
    │
    ▼
Detection Result
```

Confidence Threshold와 IoU Threshold를 설정하여 detection 결과를 filtering하며, PyTorch 및 ONNX 모델을 선택적으로 사용할 수 있도록 구성했습니다.

---

## ROS-based Real-time Pipeline

얼굴 탐지 모델을 단순 이미지 inference가 아닌 **ROS Node 기반 실시간 Pipeline**으로 구성했습니다.

### FPS Controller

카메라에서 입력되는 모든 Frame을 그대로 처리할 경우 Edge Device의 추론 속도보다 입력 Frame Rate가 높아 지연이 누적될 수 있습니다.

이를 완화하기 위해 별도의 `FPS Controller` Node를 구성하여 입력 Frame을 선택적으로 전달합니다.

```text
Camera Node
     │
     ▼
/camera/color/image_raw
     │
     ▼
FPS Controller
     │
     ▼
/fps_controller/image_raw
```

현재 구현에서는 입력되는 Frame 중 일정 간격의 Frame만 다음 Node로 전달하도록 구성되어 있습니다.

### Face Detector

Face Detector Node는 FPS Controller로부터 영상을 subscribe하고 YOLOv6 inference를 수행합니다.

```text
/fps_controller/image_raw
          │
          ▼
     Face Detector
          │
          ▼
 YOLOv6 Inference
          │
          ▼
/face_detector/image_result
```

탐지된 얼굴 Bounding Box를 포함한 결과 영상을 ROS Image Message로 변환하여 publish합니다.

---

## Edge Inference

Edge Device에서의 실시간 실행을 고려하여 다양한 inference 방식을 실험할 수 있도록 구성했습니다.

* **PyTorch Inference**
* **ONNX Export / Runtime**
* **TensorRT Inference**
* **FP16 Inference**

이를 통해 모델 정확도뿐만 아니라 실제 시스템 환경에서의 **Latency와 Throughput을 고려한 inference pipeline**을 구성했습니다.

---

## Utilities

### Face Extraction

YOLOv6에서 탐지한 Bounding Box를 기반으로 얼굴 영역을 추출하여 별도의 이미지로 저장할 수 있습니다.

추출된 얼굴 이미지는 Gallery 구성이나 후속 Face Recognition 모델의 입력 데이터로 활용할 수 있습니다.

### Face Mosaic

탐지된 얼굴 영역에 Mosaic 처리를 적용하는 기능을 제공합니다.

### Dataset & Evaluation

얼굴 탐지 모델 학습 및 평가를 위한 데이터 처리와 inference utility를 포함합니다.

* Dataset preprocessing
* YOLOv6 training
* Detection evaluation
* ONNX export
* TensorRT evaluation

---

## Repository Structure

```text
ROS_face_detect/
├── configs/                    # YOLOv6 model configuration
├── data/                       # Dataset configuration
├── tools/                      # Training / inference utilities
├── yolov6/                     # YOLOv6 implementation
│
├── core_train.py               # YOLOv6 training
├── core_eval.py                # Model evaluation
│
├── face_detector.py            # ROS face detector node
├── fps_controller.py           # Input frame rate controller
├── video_publisher.py          # ROS video publisher
├── video_subscriber.py         # ROS video subscriber
│
├── Face_extractor.py           # Detected face extraction
├── Face_mosaic.py              # Face mosaic processing
│
├── onnx_export.py              # ONNX model export
├── onnx_runtime_dynamic_batch.py
├── tensorrt_eval.py            # TensorRT inference / evaluation
│
└── requirements.txt
```

---

## Key Features

* **YOLOv6-based Face Detection**
  YOLOv6를 활용한 실시간 얼굴 Bounding Box 탐지

* **ROS Integration**
  ROS Publisher / Subscriber 기반 영상 처리 Pipeline 구성

* **Frame Rate Control**
  Edge Device의 처리량을 고려한 입력 Frame 제어

* **Edge Inference Optimization**
  ONNX, TensorRT, FP16 기반 inference 지원

* **Face Processing Utilities**
  탐지된 얼굴의 추출 및 Mosaic 처리

---

## Tech Stack

`Python` · `PyTorch` · `YOLOv6` · `ROS` · `OpenCV` · `ONNX` · `TensorRT` · `CUDA`
