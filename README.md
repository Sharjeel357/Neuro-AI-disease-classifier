# NeuroAI — Neurological Disease Classifier

> An AI-powered deep learning application for classifying neurological conditions from brain images using trained CNN models and an interactive Gradio interface.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=for-the-badge\&logo=gradio\&logoColor=white)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-FFD21E?style=for-the-badge\&logo=huggingface\&logoColor=black)](https://huggingface.co/spaces/sharjeel357/NeuroAI)

---

## Overview

**NeuroAI** is a computer vision and deep learning project designed to classify neurological conditions from brain medical images.

The application uses trained **PyTorch neural network models** and provides a simple **Gradio web interface** where users can upload an image and receive the model's predicted class.

The project demonstrates an end-to-end AI workflow:

**Image → Preprocessing → CNN Model → Classification → Prediction**

The trained models are integrated directly into the application, making the project easy to run locally or deploy as an interactive AI application.

---

## Features

* Brain image classification using deep learning
* Multiple trained PyTorch `.pth` models
* Image upload interface
* Interactive Gradio web application
* Computer vision-based prediction pipeline
* Python-based implementation
* Deployed on Hugging Face Spaces
* Can be run locally
* Lightweight project structure

---

## Project Architecture

```text
                         ┌─────────────────────┐
                         │      User Image     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Image Preprocessing │
                         │  Resize / Normalize │
                         └──────────┬──────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────┐
                  │        Trained CNN Models        │
                  │                                  │
                  │  • Alzheimer Model               │
                  │  • Brain Tumor Model             │
                  │  • Brain Stroke Model            │
                  │  • Brain Hemorrhage Model        │
                  └───────────────┬──────────────────┘
                                  │
                                  ▼
                         ┌─────────────────────┐
                         │ Model Prediction    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Gradio Web Interface│
                         └─────────────────────┘
```

---

## Models

The repository contains trained PyTorch model weights for multiple neurological and brain-related classification tasks:

| Model                           | File                |
| ------------------------------- | ------------------- |
| Alzheimer Classification        | `alzmodel.pth`      |
| Brain Hemorrhage Classification | `brainham.pth`      |
| Brain Stroke Classification     | `brainst_model.pth` |
| Brain Tumor Classification      | `braintmodel.pth`   |

> **Note:** The models are intended for educational and research purposes and should not be considered a substitute for professional medical diagnosis.

---

## Tech Stack

### Programming

* Python

### Machine Learning and Deep Learning

* PyTorch
* Convolutional Neural Networks (CNN)
* Image Classification
* Computer Vision

### Application

* Gradio

### Deployment

* Hugging Face Spaces

### Model Format

* PyTorch `.pth` model weights

---

## Project Structure

```text
Neuro-AI-disease-classifier/
│
├── app.py                  # Gradio application
├── requirements.txt        # Python dependencies
│
├── alzmodel.pth            # Alzheimer model
├── brainham.pth            # Brain hemorrhage model
├── brainst_model.pth       # Brain stroke model
├── braintmodel.pth         # Brain tumor model
│
├── README.md
└── .gitattributes
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sharjeel357/Neuro-AI-disease-classifier.git
cd Neuro-AI-disease-classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Gradio will provide a local URL in the terminal.

Open the URL in your browser and upload an image to test the classifier.

---

## Live Demo

Try NeuroAI on Hugging Face Spaces:

https://huggingface.co/spaces/sharjeel357/NeuroAI

---

## How It Works

### 1. Image Input

The user uploads a brain image through the Gradio interface.

### 2. Preprocessing

The input image is prepared according to the preprocessing expected by the corresponding trained model.

### 3. Model Selection

The application uses the appropriate trained model for the selected classification task.

### 4. Inference

The image is passed through the trained CNN model.

### 5. Classification

The model produces a prediction for the corresponding class.

### 6. Result

The prediction is displayed through the Gradio interface.

---

## Machine Learning Pipeline

```text
Raw Brain Image
      │
      ▼
Image Loading
      │
      ▼
Preprocessing
      │
      ├── Resize
      ├── Normalize
      └── Tensor Conversion
      │
      ▼
CNN Model
      │
      ▼
Feature Extraction
      │
      ▼
Classification Layer
      │
      ▼
Predicted Class
```

---

## Project Goals

The primary goals of NeuroAI are to:

* Explore the application of deep learning to medical image classification.
* Build practical computer vision models using PyTorch.
* Develop an interactive AI application rather than only a standalone model.
* Demonstrate model deployment using Gradio and Hugging Face Spaces.
* Create an end-to-end machine learning project suitable for experimentation and research.

---

## Future Improvements

* [ ] Improve model accuracy and generalization
* [ ] Add confidence scores to predictions
* [ ] Add Grad-CAM visualizations for model interpretability
* [ ] Add a unified multi-disease model
* [ ] Improve image preprocessing and augmentation
* [ ] Add model evaluation metrics
* [ ] Add confusion matrices and classification reports
* [ ] Add explainable AI features
* [ ] Improve UI/UX
* [ ] Add automated model selection
* [ ] Add additional neurological conditions
* [ ] Containerize the application using Docker

---

## Disclaimer

**NeuroAI is an educational and research-oriented project.**

The predictions generated by this application should **not** be interpreted as medical advice, diagnosis, or treatment recommendations.

Medical imaging should always be evaluated by qualified healthcare professionals using appropriate clinical procedures and validated diagnostic systems.

---

## Author

### Sharjeel Hashmi

Computer Science Engineer | AI/ML | Deep Learning | Generative AI

GitHub:
https://github.com/Sharjeel357

Hugging Face:
https://huggingface.co/sharjeel357

---

## Support

If you find this project interesting or useful, consider giving the repository a star on GitHub.

---

## License

This project is intended for educational and research purposes.

Please review the dataset and model licensing requirements before using the trained models or associated datasets for commercial applications.
