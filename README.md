# car-type-classification
Deep learning project for fine-grained car type classification using Stanford Cars dataset. Includes training of ResNet50, InceptionV3, EfficientNetB0, evaluation (accuracy, confusion matrix, Grad-CAM), and a Streamlit GUI for predictions
🚗 Car Type Classification — Stanford Cars Project

This repository contains a complete deep learning pipeline for fine-grained car type classification using the Stanford Cars Dataset.
The project includes data preprocessing, training three transfer-learning models, evaluation & Grad-CAM visualization, and a full Streamlit Web App.

---

📌 Project Goal

Classify car images into their make/model/year (196 classes) and compare the performance of three CNN architectures:

- ResNet50
  
- InceptionV3
  
- EfficientNetB0

---

📂 Repository Structure

/ (root)
├── README.md
├── TEAM.md
├── data/
│   └── download_data.sh
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── gradcam.py
│   └── gui/
│       └── app.py
├── results/
│   ├── confusion_matrix.png
│   └── gradcam_examples/
└── docs/
    └── evaluation_report.pdf

---

🧠 Models Used

We compare three transfer-learning architectures:

Model| Notes
ResNet50| Strong CNN using residual blocks
InceptionV3| Extracts multi-scale details via inception blocks
EfficientNetB0| Lightweight and fast

Each model is trained on:

- Train/Validation/Test split
- Augmentation (flip, zoom, rotation)
- Learning-rate scheduling
- Early stopping

---

📊 Evaluation Metrics

Each model is evaluated using:

- Top-1 accuracy
- Confusion matrix
- Grad-CAM visualizations
- Comparison table for best model

---

📸 Streamlit GUI Features

The web app allows:

- Uploading a car image
- Showing Top-3 predictions + confidence
- Visualizing Grad-CAM heatmaps
- Showing accuracy & confusion matrix
- Selecting between the 3 models

Run GUI:

streamlit run src/gui/app.py

---

📥 Dataset

Stanford Cars Dataset (196 classes)
Includes: make / model / year.

To download:

bash data/download_data.sh

Or manually download from official link.

---

🧪 Training Example

python src/models/train.py --model resnet50 --epochs 10 --batch-size 32 --data-dir ./data

---

🧑‍💻 Team Workflow (3 Members)

We divided the work into 3 main sections:

1. Data Team

- Download & preprocess dataset
- Augmentation
- Train/val/test split

2. Model & Evaluation Team

- Train the 3 models
- Calculate accuracy + confusion matrix
- Grad-CAM visualization

3. GUI Team

- Build Streamlit app
- Connect best model
- Display predictions + Grad-CAM

Full details in "TEAM.md".

---

🛠 Tech Stack

- TensorFlow / Keras
- Python
- NumPy, Pandas, Matplotlib
- Streamlit
- Grad-CAM

---

📎 How to Contribute

1. Create a feature branch:

git checkout -b feature/<name>

2. Make your changes, then push:

git push origin feature/<name>

3. Open a Pull Request → request review → merge.

---

📄 License

This project is for educational use.

---

🎥 Demo & Results

All visual results, confusion matrices, and Grad-CAM samples are in the "results/" folder.
A short demo video can be found in "docs/" (if added).

---

👥 Team Members

Names and responsibilities are listed in TEAM.md.
