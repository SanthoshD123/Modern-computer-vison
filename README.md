# Modern Computer Vision Repository

This repository contains a comprehensive collection of Jupyter notebooks covering a wide range of computer vision techniques and deep learning applications for image and video processing. It serves as both a learning resource and a reference implementation for various computer vision tasks.

## Repository Structure

The repository is organized into two main directories:

### 1. Deep Learning CV

Contains notebooks focused on deep learning approaches to computer vision, including:

- **Convolutional Neural Networks (CNNs)**
  - CNN Visualizations (Filters, Activations, GradCAM)
  - Transfer Learning with pre-trained models
  - Performance analysis and model fine-tuning

- **Image Classification**
  - Cats vs Dogs classification with PyTorch
  - Fashion-MNIST implementations with regularization techniques

- **Generative Models**
  - GANs (Generative Adversarial Networks)
  - CycleGAN for image translation (Horses to Zebras)
  - ArcaneGAN for artistic style transfer
  - StyleGAN for anime generation

- **Image Segmentation**
  - DeepLabV3
  - Mask R-CNN
  - U-Net and SegNet architectures

- **Object Detection**
  - YOLOv3, YOLOv4, and YOLOv5
  - Faster R-CNN implementations
  - SSD (Single Shot Detector) with MobileNetV2
  - Custom detectors for specific applications (Chess pieces, maritime objects, potholes)

- **Web Applications**
  - Flask REST API (Client and Server implementations)
  - Flask web applications for computer vision services

### 2. OpenCV

Contains notebooks demonstrating fundamental to advanced OpenCV techniques:

- **Image Processing Fundamentals**
  - Loading, displaying, and saving images
  - Color filtering and grayscaling
  - Arithmetic and bitwise operations
  - Transformations, translations, and rotations

- **Feature Detection and Analysis**
  - Contour detection and analysis
  - Edge and corner detection
  - Face and eye detection with Haar cascades
  - Facial landmarks and recognition with Dlib

- **Video Processing**
  - Webcam access and video capture
  - Video streaming (RTSP, IP cameras)
  - Motion tracking with optical flow
  - Object tracking by color

- **Advanced Techniques**
  - Background/foreground subtraction
  - Perspective transforms
  - Watershed algorithm for segmentation
  - GrabCut for background removal

## Frameworks and Libraries

The notebooks in this repository use various frameworks and libraries, including:

- **PyTorch**: For deep learning models and neural networks
- **Keras/TensorFlow**: Alternative deep learning implementations
- **OpenCV**: For computer vision algorithms and image processing
- **Dlib**: For facial landmark detection and face recognition
- **Flask**: For creating web services and applications

## Getting Started

1. Clone this repository
2. Install the required dependencies (consider using a virtual environment)
3. Navigate to the notebook of interest and run it in Jupyter

## Requirements

The notebooks require Python 3.6+ and the following main packages:
- pytorch
- tensorflow
- keras
- opencv-python
- numpy
- matplotlib
- dlib
- flask

A detailed `requirements.txt` file will be provided soon.

## Contributing

Feel free to contribute to this repository by adding new notebooks, improving existing ones, or fixing issues. Please make sure to follow the existing structure and coding style.

## License

[To be added]

## Acknowledgments

Special thanks to all the open-source communities behind the libraries and frameworks used in these notebooks.
