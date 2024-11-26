# Face_Emotion_Recognition_Machine_Learning
# Real-Time Facial Emotion Recognition System

This project focuses on a **real-time facial emotion recognition system** utilizing **CNN (Convolutional Neural Network)** for feature extraction and **fuzzy logic** for decision-making.

---

## Agenda
1. Problem Statement
2. Dataset Overview
3. System Architecture
4. Algorithms & Evaluation Metrics
5. Results
6. Conclusion

---

## Problem Statement

### Challenges in Emotion Recognition:
- Variability in lighting conditions (dark, bright, medium).
- Handling low-confidence predictions from CNN models.
- Improving adaptability of machine decisions in real-world scenarios.

### Objective:
Develop a robust system combining CNN for feature extraction and fuzzy logic for decision-making.

---

## Dataset Overview

- **Size & Format**: 36,000 grayscale images (48x48 pixels).
- **Emotion Classes**: Neutral, Happy, Sad, Surprise, Disgust, Fear, Angry.
- **Data Distribution**: 
  - Training Data: 28,881 images (80%)
  - Testing Data: 7,066 images (20%)

### Preprocessing Steps:
1. Convert images to grayscale.
2. Extract features into NumPy arrays.
3. Store data as training and testing datasets.

---

## System Architecture

### CNN for Feature Extraction:
- **Input Layer**: 48x48 images flattened to a 1D array (size 2304).
- **Hidden Layers**:
  - Convolutional layers: 3x3 filters with 64 and 32 neurons.
  - Max-pooling layers: Downsamples spatial dimensions.
  - Dense layers: 128 and 64 neurons for deeper features.
  - Activation Functions: 
    - **Hidden Layers**: ReLU
    - **Output Layer**: Softmax
- **Output Layer**: 7 neurons representing emotions.

### Fuzzy Logic for Decision-Making:
- **Inputs**:
  - Brightness: Categorized as Dark, Medium, Bright.
  - CNN Emotion Confidence: Low, Medium, High.
- **Rules**:
  - High confidence & Bright = **Accept**.
  - Medium confidence or Medium brightness = **Uncertain**.
  - Low confidence or Dark brightness = **Reject**.

---

## Algorithms & Evaluation Metrics

### CNN:
- Image processing and feature extraction.
- Achieved **73.34% accuracy** after 100 epochs.

### Fuzzy Logic:
- Fine-tunes predictions based on lighting and CNN confidence.

---

## Results
- Emotion recognition system effectively identifies seven emotions.
- Fuzzy logic enhances CNN predictions by addressing uncertainties.
- 
<img width="787" alt="Screenshot 2024-11-25 at 4 05 41 PM" src="https://github.com/user-attachments/assets/7dd23763-5cff-4d13-a7e8-1389cec18da1">

<img width="763" alt="Screenshot 2024-11-25 at 4 08 47 PM" src="https://github.com/user-attachments/assets/ec2c3b54-2629-4a52-9194-d8ca6c19f801">

---<img width="966" alt="Screenshot 2024-11-25 at 4 51 31 PM" src="https://github.com/user-attachments/assets/ec31fd49-d38f-46d3-9d9a-824e475aae44">

<img width="966" alt="Screenshot 2024-11-25 at 4 52 38 PM" src="https://github.com/user-attachments/assets/2831780e-cbf9-4112-b488-6505cdd7672c">
<img width="960" alt="Screenshot 2024-11-25 at 4 24 23 PM" src="https://github.com/user-attachments/assets/efb4bc2b-a15f-4aa6-8ef8-ffab1e43bb50">

## Conclusion

### Summary:
- The system integrates CNN and fuzzy logic to achieve robust emotion recognition.
- Adaptable to varying lighting and prediction confidence.

### Future Scope:
- Enhance accuracy using larger datasets.
- Explore real-time applications like:
  - Customer engagement systems.
  - Mental health monitoring.

---

## Contributions
This project combines state-of-the-art algorithms with decision-making adaptability to address real-world challenges in emotion recognition.
