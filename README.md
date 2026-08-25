# 🎓 Next-Gen Admission Predictor

A Streamlit-based machine learning web application that predicts university admission probability using an **XGBoost Classifier**.

The application provides real-time admission probability updates based on an applicant's GRE score, TOEFL score, CGPA, SOP strength, LOR strength, and research experience.

## 🚀 Features

* 🎯 Real-time admission probability prediction
* 📊 XGBoost classification model
* 📈 Model performance evaluation
* 🔥 Confusion matrix
* 📉 ROC curve and AUC
* 📋 Classification report
* 📊 Feature importance analysis
* 📈 Learning curve
* 📊 Data distribution analysis
* 🔗 Feature correlation heatmap
* 🏫 University-wise admission probability simulation
* 💡 Personalized suggestions for improving admission chances

## 🧠 Machine Learning Model

The application uses an **XGBoost Classifier** trained on a synthetically generated dataset of 1,000 samples. The dataset contains:

* GRE Score
* TOEFL Score
* CGPA
* SOP Strength
* LOR Strength
* Research Experience

The data is split into **800 training samples and 200 test samples**.
The model uses the following main parameters:

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.9
colsample_bytree = 0.9
```

## 📊 Applicant Inputs

| Feature             | Range    |
| ------------------- | -------- |
| GRE Score           | 260–340  |
| TOEFL Score         | 80–120   |
| CGPA                | 6.0–10.0 |
| SOP Strength        | 1–5      |
| LOR Strength        | 1–5      |
| Research Experience | Yes / No |

## 🖥️ Application Sections

### 🎯 Prediction

Displays:

* Applicant profile summary
* Percentile ranks
* Profile radar chart
* Real-time admission probability
* Admission probability gauge
* Improvement suggestions
* University-wise admission probability simulation

### 📊 Model Performance

Displays:

* Accuracy
* Precision
* Recall
* Confusion matrix
* ROC curve
* AUC
* Classification report

### 📈 Visual Analytics

Displays:

* XGBoost feature importance
* Learning curve
* Feature distributions
* Feature correlation heatmap

### ℹ️ About

Provides information about the model, dataset, features, and application usage.

## 🛠️ Technologies Used

* Python
* Streamlit
* NumPy
* Pandas
* Matplotlib
* Seaborn
* XGBoost
* Scikit-learn
* Plotly

## 📁 Project Structure

```text
project/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Navigate into the project directory:

```bash
cd project
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📌 Important Note

This project uses a **synthetically generated dataset** inside the application rather than an external admission dataset. The admission labels are generated using predefined criteria with a small amount of noise.

Therefore, the predicted admission probabilities are intended for **educational and demonstration purposes** and should not be treated as actual university admission decisions.

## 👩‍💻 Project

**Next-Gen Admission Predictor**

An interactive application built with Streamlit and XGBoost to explore university admission prediction through machine learning.
