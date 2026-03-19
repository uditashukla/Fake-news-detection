#  Fake News Detection using Machine Learning

##  Project Overview

This project is a **Fake News Detection System** built using **Machine Learning and Natural Language Processing (NLP)**.
It classifies news articles as **Fake or Real ** based on their content.

The system uses text preprocessing techniques and a trained model to identify whether a news headline is trustworthy or not.

---

##  Features

* Classifies news into Fake or Real
* NLP-based text preprocessing
* TF-IDF vectorization (with n-grams)
* Machine Learning model (Random Forest)
* End-to-end pipeline (data → model → prediction)

---

##  Tech Stack

* Python 
* Pandas & NumPy
* NLTK (Natural Language Processing)
* Scikit-learn (Machine Learning)

---

## Dataset

The dataset consists of two CSV files:

* `Fake.csv` → Fake news articles
* `True.csv` → Real news articles

Each dataset contains:

* Title
* Text
* Date

---

##  Project Workflow

### 1. Data Loading

* Load Fake and True datasets
* Add labels:

  * 0 → Fake News
  * 1 → Real News

---

### 2. Data Preprocessing

* Remove special characters
* Convert text to lowercase
* Remove stopwords
* Apply lemmatization

---

### 3. Feature Extraction

* TF-IDF Vectorizer
* Max features: 5000
* N-grams: (1, 3)

---

### 4. Model Training

* Train-test split (70:30)
* Model used: Random Forest Classifier

---

### 5. Model Evaluation

* Accuracy Score
* Classification Report
* Confusion Matrix

---

##  Sample Output

Input:

```
Breaking news: Government announces new economic policy
```

Output:

```
 Real News
```

---

##  Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### Step 2: Install Dependencies

```bash
pip install pandas numpy nltk scikit-learn
```

### Step 3: Run the Project

```bash
python your_script_name.py
```

---

##  Project Structure

```
fake-news-detection/
│── Fake.csv
│── True.csv
│── your_script_name.py
│── news_dataset.csv
│── README.md
```

---

##  Future Improvements

* Try advanced models (Logistic Regression, Naive Bayes, LSTM)
* Improve accuracy using deep learning
* Add real-time news data
* Deploy as web application

---

##  Contributing

Feel free to fork this repository and contribute to improve the project.

---

