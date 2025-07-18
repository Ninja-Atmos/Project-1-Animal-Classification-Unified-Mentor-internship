
# 🐾 Animal Image Classification

> **Internship Project**  
> Deep learning–based image classification system to recognize animals in images using Transfer Learning.

---

## 📋 **Table of Contents**
- [Objective](#objective)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Libraries & Tools Used](#libraries--tools-used)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Future Work](#future-work)
- [Author](#author)

---

## 🎯 **Objective**
The aim of this internship project was to build a robust image classification system capable of identifying animals from images.  
The system classifies an input image into one of **15 predefined animal classes**.

---

## 🐘 **Dataset**

📥 **Download Link:** [Google Drive](https://drive.google.com/file/d/15I_LqEvJL9ktEi6khTyulMHLO_3KiTnO/view?usp=sharing)
- The provided dataset contains images organized into **15 folders**, each corresponding to one animal class.  
- Each image is an RGB image of size **224×224×3**, suitable for deep learning models.  
- Data was split into:
  - **Training set:** 80%
  - **Validation set:** 20%

### 📂 **Classes:**
`Bear`, `Bird`, `Cat`, `Cow`, `Deer`, `Dog`, `Dolphin`, `Elephant`, `Giraffe`, `Horse`, `Kangaroo`, `Lion`, `Panda`, `Tiger`, `Zebra`

---

## 🧠 **Approach**
✅ Data exploration and cleaning: ensured all class folders contain valid images  
✅ Split dataset into training & validation subsets  
✅ Implemented **Transfer Learning** using a pre-trained MobileNetV2 as feature extractor  
✅ Trained a custom classification head on the dataset  
✅ Evaluated model performance using standard classification metrics and visualizations

---

## 🏗️ **Model Architecture**
- **Base Model:** MobileNetV2 (`imagenet` weights, frozen during initial training)
- **Custom Head:**
  - Global Average Pooling
  - Dense layer with 256 units, ReLU activation
  - Dropout layer with rate 0.5
  - Output Dense layer with 15 units, Softmax activation

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  
**Metrics:** Accuracy  
**Epochs:** 10 (configurable)

---

## 🧰 **Libraries & Tools Used**
The following libraries and tools were utilized in this project:
- 🐍 **Python 3.8+**
- 🔷 **TensorFlow** – for building and training deep learning models
- 🔷 **Keras** (integrated with TensorFlow) – for high-level neural network API
- 📊 **scikit-learn** – for train/test splitting, metrics, and evaluation
- 📊 **matplotlib** & **seaborn** – for visualizing training curves and confusion matrix
- 📂 **pathlib**, **os**, **shutil** – for filesystem and data management
- 📦 **Google Colab** – cloud-based GPU-accelerated training environment

---

## 📊 **Results**
✅ Training & Validation accuracy and loss were tracked and plotted over epochs.  
✅ Generated a detailed **classification report** showing precision, recall, and F1-score per class.  
✅ Visualized a **confusion matrix** to analyze per-class performance.

---

## 🚀 **How to Run**
### Option 1: On Google Colab
✅ Upload `animal_classification.zip` and run the provided `colab_notebook.ipynb`.  
✅ Execute each cell step-by-step to train and evaluate the model.

### Option 2: Locally
1️⃣ Install required dependencies:
```bash
pip install tensorflow scikit-learn matplotlib seaborn
```
2️⃣ Run the notebook or Python script in your local Python environment.

---

## 📦 **Dependencies**
- Python ≥ 3.7
- TensorFlow ≥ 2.8
- scikit-learn
- matplotlib
- seaborn

These can be installed using `pip install -r requirements.txt` (optional).

---

## 🌟 **Future Work**
✨ Fine-tune base model layers to improve accuracy  
✨ Experiment with other architectures (e.g., ResNet50, EfficientNet, Vision Transformers)  
✨ Deploy the trained model as an interactive web application (e.g., Streamlit, Flask)  
✨ Implement more sophisticated data augmentation or synthetic data generation  
✨ Extend the dataset to include more animal classes or higher-resolution images

---

## 👨‍💻 **Author**
**Suvam Biswas**  
*Machine Learning Intern*
