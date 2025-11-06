# Aerial Object Classification & Detection
This project aims to develop a **deep learning-based solution** that classifies aerial images into **Bird** or **Drone**

## Objectives
- Build and compare **Custom CNN** and **Transfer Learning** models.
- Deploy the final app using **Streamlit**.

## Skills Learned
Deep Learning • Computer Vision • TensorFlow/Keras • Streamlit • Transfer Learning

## Dataset Sources
**Classification Dataset:** Bird/Drone images. Google Drive Link: https://drive.google.com/drive/folders/1DE7FY5LwUoCIVgJqSeJc89q0Tvii-C51?usp=drive_link

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
| EfficientNetB0 | 0.986 | 0.989 | 0.979 | 0.984 | 

## Conclusion
**The EfficientNetB0 model classifies images more accurately with average confidence of: 0.9821**

## Live Demo
Try the app here: [https://aerial-classification.streamlit.app/](https://aerial-classification.streamlit.app/)

## User Interface:
<img width="1920" height="1080" alt="image (2)" src="https://github.com/user-attachments/assets/2288d9cc-9cde-4a92-b6ac-e879c9aeeaa9" />
<img width="1920" height="1080" alt="image (1)" src="https://github.com/user-attachments/assets/35f454f5-611f-45fa-a3a7-040cec89432b" />
<img width="1920" height="1080" alt="image (7)" src="https://github.com/user-attachments/assets/57c9e06b-fc7a-417f-ba9a-e6a234df3cdc" />
<img width="1920" height="1080" alt="image (5)" src="https://github.com/user-attachments/assets/dfeead76-8b9b-41fd-ab7a-52f25967b15a" />


## Run the Streamlit App
```bash
streamlit run app.py
