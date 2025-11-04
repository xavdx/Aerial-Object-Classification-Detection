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
| Custom CNN | Best performing binary classifier | [Google Drive Link](https://drive.google.com/file/d/1an3dORosSu-L2u4ZYY61wXgjV85cpRDG/view?usp=drive_link) |
| EfficientNetB0 | Transfer Learning Model | [Google Drive Link](https://drive.google.com/file/d/1JaESQDhSdSJD0kgfPIHajyU30P9Ad5eq/view?usp=sharing) |

## Results & Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Custom CNN | 0.781 | 0.724 | 0.809 | 0.764 |
| EfficientNetB0 | 0.521 | 0.471 | 0.787 | 0.590 | 

## Run the Streamlit App
```bash
streamlit run app.py
