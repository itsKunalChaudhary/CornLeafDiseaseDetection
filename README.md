# 🌿 Corn Leaf Disease Detection using CNN and Transfer Learning

This project aims to classify plant leaf images into four disease categories using Convolutional Neural Networks (CNN) and Transfer Learning with **ResNet50** and **VGG16**. The dataset is divided into training, validation, and test sets, and models are evaluated using metrics such as accuracy, confusion matrix, and ROC-AUC curves.

---

## 📂 Dataset Structure

The dataset is expected to be in the following format:
```
/data
  /class_1
    image1.jpg
    image2.jpg
    ...
  /class_2
    ...
```

---

## 📌 Key Steps and Highlights

### ✅ 1. **Data Preparation**
- Loaded images from structured folders.
- Split into **81% training**, **9.5% validation**, and **9.5% test**.
- Visualized image distribution and sample images.

### ✅ 2. **Data Generators**
Used `ImageDataGenerator` to feed images for training and validation with:
- `target_size`: (224, 224)
- `color_mode`: 'rgb'
- `class_mode`: 'categorical'

---

## 🧠 CNN Model (Custom Built)

### 🏗️ Architecture
- 5 Convolution Blocks with 64 filters.
- Batch Normalization and Dropout layers.
- Fully connected layers with L2 regularization.
- Output: `softmax` layer for 4 classes.

### 🛑 Early Stopping
Custom callback halts training after reaching **98% validation accuracy**.

### 📈 Results
- **Train Accuracy:** ~99%
- **Validation Accuracy:** ~98%
- **Test Accuracy:** ~97%
- Classification Report and Confusion Matrix indicate **high precision and recall** across all classes.

---

## 🏆 ROC & AUC Evaluation

- Computed ROC curves and AUC scores for each class.
- Achieved **Micro-average AUC ~0.99**, **Macro-average AUC ~0.99**.

---

## 🔁 Transfer Learning with ResNet50

### ⚙️ Setup
- Used pre-trained `ResNet50` (weights from ImageNet).
- Added Global Average Pooling, Dense, and Dropout layers.
- Fine-tuned top layers.

### 📈 Performance
- Accuracy on test set: **~97%**
- Micro-average AUC: **0.99**

---

## 🔁 Transfer Learning with VGG16 (Not shown in full)
VGG16 architecture was set up similarly to ResNet50:
- Pretrained base frozen initially.
- Custom classifier layers added.
- Training and evaluation conducted similarly.

---

## 🧪 Model Comparison Summary

| Model       | Test Accuracy | Micro-AUC | Macro-AUC |
|-------------|----------------|-----------|-----------|
| CNN (Custom)| ~97%          | ~0.99     | ~0.99     |
| ResNet50    | ~97%          | ~0.99     | ~0.99     |
| VGG16       | TBD           | TBD       | TBD       |

---

## ✅ Conclusions

- A **deep custom CNN** can achieve excellent performance if properly regularized.
- **Transfer Learning models** like ResNet50 match or slightly exceed performance while training faster.
- **ROC-AUC** provides a clearer view of model generalization than just accuracy.
- The dataset quality and clear labeling were critical in achieving high model performance.

---

## 💾 Model Saving
Models are saved in `.h5` format for reuse:
```python
model.save('model2.h5')
```

---

## 📌 Requirements

Install dependencies with:
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python pillow
```

---

## 🚀 How to Run

1. Place dataset in `archive (1)/data/`.
2. Run the script from your IDE or terminal.
3. Saved model and visualizations will be output.

---

## 📬 Feedback

For suggestions or improvements, feel free to open an issue or pull request!
