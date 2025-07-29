# 🧪 Toxic Comment Classifier using BERT

This repository contains a fine-tuned **binary classification model** built using `bert-base-uncased` to detect **toxic vs. non-toxic comments**. The model is trained on a publicly available dataset using the Hugging Face `transformers` and `datasets` libraries with **PyTorch** backend.

---

## 🔍 Overview

- **Model**: `bert-base-uncased`
- **Task**: Binary Text Classification (Toxic vs. Non-Toxic)
- **Training Framework**: Hugging Face `Trainer` API
- **Language**: English
- **Backend**: PyTorch (TensorFlow disabled)
- **Inference API**: Hugging Face `pipeline`

---

## 📂 Dataset

Dataset used: [`llangnickel/long-covid-classification-data`](https://huggingface.co/datasets/llangnickel/long-covid-classification-data)

📌 *Note: Although originally intended for COVID-related sentiment, this code structure supports any binary classification task (e.g., toxic comment detection) with `text` and `label` fields.*

---

## 🧠 Model Architecture

- Pretrained: `bert-base-uncased` from Hugging Face Hub
- Fine-tuned on tokenized text using standard classification head
- Max token length: `512`

---

## 🚀 How to Run

### 1. Clone the repo

git clone https://github.com/your-username/toxic-comment-classifier.git
cd toxic-comment-classifier


### 2. Install Requirements

pip install transformers datasets scikit-learn torch

### 3. Train Model

python toxic_train.py  # Assuming the training code is inside this file

### 4. Tech Stack

🤗 Hugging Face Transformers
🤗 Datasets
PyTorch
Scikit-learn
BERT

## License

This project is released under the MIT License.

