## E-Waste Image Classification Using Transfer Learning

## Overview
This project demonstrates a Deep Learning–based image classification system that categorizes different types of Electronic Waste (E-Waste) using Transfer Learning.  
A pre-trained EfficientNet model is fine-tuned on a custom E-waste image dataset to automate waste segregation for recycling and environmental sustainability.

---

## Key Features
- Uses Transfer Learning & Deep Learning
- Classifies E-Waste into multiple categories
- End-to-end Machine Learning workflow:
  - Dataset preparation & preprocessing
  - Data augmentation
  - Model training using EfficientNet
  - Evaluation & visualization
  - Model saving and prediction

---

## Tech Stack
- Language: Python
- Platform: Google Colab

---

## Libraries Used
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- PIL
- Scikit-learn
- Gradio (optional demo application)

---

## Model
- Architecture: EfficientNet (Transfer Learning)
- Pretrained on: ImageNet
- Fine-tuned final layers for E-waste classification
- Global Average Pooling + Dense Layers
- Softmax output layer for multi-class classification

---

## Dataset
Custom image dataset containing E-waste categories:
- Battery
- Mobile
- Keyboard
- PCB
- Printer
- Television
- Washing Machine
- Other electronics

---

## Dataset Structure
- Training set
- Testing / Validation set

---

## How to Run
1. Open the `.ipynb` file in Google Colab
2. Upload or connect the image dataset
3. Update the dataset path if required
4. Run all cells in sequence to:
   - Load the dataset
   - Train the model
   - Evaluate accuracy
   - Save the trained model
   - Test predictions on sample images

---

## Output
The model:
- Predicts the E-waste category
- Displays confidence scores
- Shows performance graphs:
  - Training & validation accuracy
  - Training & validation loss

---

## Optional
- Gradio interface for interactive predictions

---

## Use Case
This project demonstrates:
- Practical application of Transfer Learning
- AI for sustainability & smart waste management
- Academic and research-oriented ML implementation

The model classifies E-waste images with good prediction accuracy based on the trained dataset.

---

## Author
Manickavel Palani