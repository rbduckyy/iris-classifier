# Iris Classifier — Beginner Machine Learning Project

Loads `iris.csv`, performs EDA, visualizes features, trains Logistic Regression and k‑NN classifiers, and saves plots.

This project loads the Iris dataset, performs exploratory data analysis (EDA), visualizes feature relationships, trains two beginner machine‑learning models, and generates evaluation plots like a confusion matrix.

It was created as part of a skill‑building portfolio for future undergraduate **ARCH research** in engineering, data analysis, and computational modeling.

---

## 📁 Project Structure

```
iris-classifier/
│
├── iris_classify.py          # Main script
├── iris.csv                  # Dataset (150 samples, 3 species)
├── requirements.txt          # Python dependencies
├── scatter_sepal.png         # Scatter plot (sepal features)
├── pairplot.png              # Multivariate pairplot
├── confusion_matrix_knn.png  # Confusion matrix for k-NN
└── README.md                 # This file
```

## Dataset Overview

### **The Iris dataset contains **150 samples** belonging to **3 species**:**
- *Iris setosa*  
- *Iris versicolor*  
- *Iris virginica*

### **Each sample includes:**
- Sepal length  
- Sepal width  
- Petal length  
- Petal width  

---

## Features of This Project

### **1. Exploratory Data Analysis (EDA)**
- Prints shape, info, and descriptive statistics  
- Detects missing values  
- Shows pairwise relationships  

### **2. Visualizations** (auto‑saved)
- `scatter_sepal.png`  
- `pairplot.png`  
- `confusion_matrix_knn.png`  

### **3. Models Implemented**
Trained on standardized features:

- **Logistic Regression**  
- **k‑Nearest Neighbors (k=5)**  

Outputs include:
- Accuracy  
- Classification report  
- Confusion matrix  

### **4. Reproducibility**
All dependencies recorded in `requirements.txt`.

---

## How to Run This Project

### **1. Clone the repo (or download ZIP)**
 - Plain Textgit clone https://github.com/<your-username>/iris-classifiercd iris-classifierShow more lines
### **2. Create a virtual environment (recommended)**
 - Windows: Shellpython -m venv .venv.\.venv\Scripts\activateShow more lines
### **3. Install dependencies**
 - Shellpip install -r requirements.txtShow more lines
### **4. Run the classifier**
 - Shellpython iris_classify.pyShow more lines
 - Plots will pop up and be saved to the project folder.

## Example Outputs
### **Scatter Plot — Sepal Length vs. Width**
 - (scatter_sepal.png)
 - scatter_sepal.png
### **Pairplot — Multivariate Relationships**
 - (pairplot.png)
 - pairplot.png
### **Confusion Matrix (k-NN)**
 - (confusion_matrix_knn.png)
 - confusion_matrix_knn.png

## Technologies Used
### Python 3.12
### pandas
### numpy
### matplotlib
### seaborn
### scikit-learn

## Contact
### If you have questions:
 - Name: Revanth Naidu Naidu
 - Email: revanth242008@gmail.com, rnaid002@fiu.edu
