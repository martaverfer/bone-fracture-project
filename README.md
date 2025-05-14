# Bone Fracture Detection in X-Ray Images Using Pre-Trained CNNs

<p align="center">
  <img src="images/StreamlitApp.png" alt="X-ray Preview" width="900"/>
</p>

## Overview

Fractures are a common clinical problem, and **radiographic imaging (X-rays)** remains the most accessible and widely used diagnostic tool for detecting them. Manual interpretation of X-rays can be time-consuming and subject to human error, especially when the fracture is subtle or the image quality is poor. 

This project aims to automate the **binary classification of bone X-rays** into:
- **Fractured**
- **Not Fractured**

Using **deep learning and transfer learning** techniques. We leverage the power of pre-trained convolutional neural networks to achieve high accuracy in fracture detection from raw X-ray images. Additionally, a **Streamlit web application** was developed to make the model predictions interactive and accessible, allowing users to upload X-ray images and receive predictions in real time.

## Dataset

**📁 Source**: [Fracture Multi-Region X-ray Dataset on Kaggle](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data)

- Over **10,000 images** across **7 body regions**:
  - Elbow, Finger, Forearm, Hand, Humerus, Shoulder, Wrist
- Split into `train`, `validation`, and `test` sets
- Each subset contains images categorized as:
  - `fractured`
  - `not fractured`

Images vary in size, contrast, and orientation, which makes **data preprocessing and augmentation** essential.

## 🧠 Deep Learning Models Used

We implemented and compared two powerful **transfer learning** architectures using **TensorFlow/Keras**:

### 1. **MobileNetV2**
-  Lightweight CNN designed for efficiency on mobile and edge devices
- Uses **depthwise separable convolutions** to reduce computation
- **~3.4 million parameters**
- Pros: Fast, memory efficient  
- Cons: Lower capacity, slightly lower accuracy on complex tasks

### 2. **ResNet50**
- Deep Residual Network with **50 layers**
- Introduced **residual (skip) connections** to ease training of deep networks
- **~25 million parameters**
- Pros: High accuracy, handles vanishing gradients well  
- Cons: More computationally expensive

> ✅ **Best results were obtained with ResNet50**, achieving **98% test accuracy**, significantly outperforming MobileNetV2.

## Workflow

Here’s a breakdown of how the pipeline is structured:

### 📁 1. Data Loading & Preprocessing
- Dataset loaded from Kaggle 
- Used `image_dataset_from_directory()` to create TensorFlow datasets

### 🔄 2. Data Augmentation
To improve generalization, especially due to varied orientations and lighting:
- Random rotation
- Zoom
- Rescaling to [0, 1]

### 🧠 3. Model Building
Two custom models were built using transfer learning:
- Load base model (MobileNetV2 or ResNet50 with `include_top=False`)
- Freeze base layers initially
- Add classification head:
  - Global Average Pooling
  - Dense layer
  - Output: Sigmoid activation for binary classification

### 🏋️‍♀️ 4. Training
- Optimizer: `Adam`
- Loss: `BinaryCrossentropy`
- Metrics: `Accuracy`
- Callbacks used:
  - `EarlyStopping`
  - `ModelCheckpoint` (saved best model)

### 📉 5. Evaluation & Visualization
- Training vs. validation accuracy/loss plots
- Confusion matrix and classification report on test set
- Sample predictions visualized with matplotlib

## 🚀 Streamlit Web Application

A user-friendly Streamlit application has been developed to facilitate the upload and prediction process. Users can:

- Upload `.jpg`, `.jpeg`, or `.png` X-ray images
- Receive real-time predictions indicating whether the bone is fractured

To run the application locally:

```bash
streamlit run app.py
```

## 📊 Results

### ResNet50 **confusion matrix** after fine tuning:

<p align="center">
  <img src="./images/confusion_matrix_resnet.png" alt="Confusion Matrix" width="600"/>
</p>

### Predictions on Test Data:

![Predictions on Test Data](images/predictions.png)

## Conclusion

This project demonstrates how **transfer learning** using state-of-the-art CNN architectures—particularly **ResNet50**—can effectively detect fractures in X-ray images with **up to 98% test accuracy**. Our approach combines model interpretability, performance, and usability.

Beyond the notebook environment, the solution was deployed through a **Streamlit app** for hands-on image upload and prediction. This interactive tool allows clinicians, students, and developers to test the model directly and observe its output—bridging the gap between model development and real-world usability.

The combination of accuracy and user accessibility makes this a strong foundation for future clinical decision support tools and showcases the real impact of deep learning in healthcare applications.

