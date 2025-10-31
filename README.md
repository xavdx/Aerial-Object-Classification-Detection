# Aerial Object Classification & Detection
This project aims to develop a **deep learning-based solution** that classifies aerial images into **Bird** or **Drone**

## Objectives
- Build and compare **Custom CNN** and **Transfer Learning** models.
- Deploy the final app using **Streamlit**.

## Skills Learned
Deep Learning • Computer Vision • TensorFlow/Keras • Streamlit • Transfer Learning

## Dataset Sources
**Classification Dataset:** Bird/Drone images  
**Object Detection Dataset:** YOLOv8 format (3319 images with .txt annotations)

## Project Workflow
1. Dataset understanding  
2. Preprocessing & augmentation  
3. Model building (Custom CNN + Transfer Learning)  
4. Training & evaluation  
5. Model comparison  
6. Streamlit deployment  

##  Download Pretrained Models
| Model | Description | Download Link |
|-------|--------------|----------------|
| Custom CNN | Best performing binary classifier | [Google Drive Link](https://drive.google.com/your_custom_cnn_model) |
| EfficientNetB0 | Transfer Learning Model | [Google Drive Link](https://drive.google.com/your_efficientnet_model) |
| YOLOv8 | Object Detection Model | [Google Drive Link](https://drive.google.com/your_yolo_model) |

## Results & Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|--------------|----------------|
| Custom CNN | 0.93 | 0.92 | 0.93 | 0.93 |
| EfficientNetB0 | 0.96 | 0.96 | 0.96 | 0.96 |

## Run the Streamlit App
```bash
streamlit run app.py
