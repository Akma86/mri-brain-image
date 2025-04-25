# Brain Tumor Segmentation and Classification

## 🎯 Project Goal
This project aims to develop a deep learning model for both **brain tumor segmentation** and **classification** using MRI images. The segmentation model is designed to identify tumor regions, while the classification model categorizes brain tumors into different types. The main goal is to assist medical professionals by automating the analysis of MRI images, enhancing diagnostic accuracy, and reducing processing time.

## 🛠 Technologies & Methods Used

### **Model**
- **Segmentation Model**: The project uses several pretrained architectures for segmentation, including **ResNet50**, **ResNet101**, **DenseNet121**, **EfficientNetB7**, and others.
- **Classification Model**: For classification, three variants of the **Swin Transformer** are used: **Swin-Tiny**, **Swin-Large**, and **Swin S3-Base**. These models are implemented using **TensorFlow** and **TensorFlowHub**.
  
### **Visualization**
- **Grad-CAM** (Gradient-weighted Class Activation Mapping) is applied to the segmentation model to generate heatmaps, highlighting the tumor areas that most influence the model's predictions.
- **Classification Visualization**: A confusion matrix and classification report are generated to evaluate model performance, alongside heatmap visualizations using **seaborn** and **matplotlib**.

### **Libraries**
- **TensorFlow**, **tf_keras**, **OpenCV**, and **tf-explain** are utilized for image processing, model training, and interpretation.

## 📊 Dataset & Results

### **Dataset**
- The dataset consists of brain MRI images with annotated tumor segmentation.
  - **Link to Classification Dataset**: [Brain Tumor Segmentation Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
  - **Link to Segmentation Dataset**: [Brain Tumor Classification MRI Dataset](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)

### **Results**
- **Segmentation**: The segmentation model demonstrates strong tumor segmentation performance, with Grad-CAM heatmaps effectively highlighting clinically relevant brain regions.
- **Classification**: The Swin Transformer models exhibit excellent performance in tumor classification, with high accuracy, precision, and recall metrics.

## 📂 Project Links
- **Brain Tumor Segmentation**: [🔗 Link](https://www.kaggle.com/code/akmalyaasir/brain-tumor-segmentations)
- **Brain Tumor Classification**: [🔗 Link](https://www.kaggle.com/code/akmalyaasir/brain-tumor-classifications)
