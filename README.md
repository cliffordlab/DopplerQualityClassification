# DopplerQualityClassification
This repository hosts the scripts used for training and deploying the deep learning model for quality classification of 1-D Doppler Ultrasound signals. The model is detailed in the paper presented at the Machine Learning for Health (ML4H) 2023 conference: [M. Motie-Shirazi, R. Sameni, P. Rohloff, N. Katebi, G. D. Clifford, "Point-of-Care Real-Time Signal Quality Assessment for Fetal Doppler Ultrasound Using a Deep Learning Approach"](https://arxiv.org/pdf/2312.09433).

## Repository Contents

### 1. `Signal_Quality_Train.py`
- This Python script includes the deep learning model training procedure for signal quality classification of Doppler recordings.

### 2. `Helper_Functions.py`
- A Python script that provides auxiliary functions essential for the training of the model.

### 3. `Model_Application_Demo.ipynb`
- A Jupyter Notebook that illustrates how to load and employ the model for classification tasks.

### 4. `saved_model` Folder
- Contains the trained deep learning model utilized for quality classification.

### 5. `Sample_Segments` Folder
- Includes a 3.75-second signal sample from each of the quality classes: 'Good', 'Poor', 'Interference', 'Talking', and 'Silent', which can be used to test the model.

