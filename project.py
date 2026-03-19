import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



def ensure_nltk_resources() -> None:
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for lookup_name, resource_name in resources.items():
        try:
            nltk.data.find(lookup_name)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_dir = Path(__file__).resolve().parent

    workspace_fake = base_dir / "Fake.csv"
    workspace_true = base_dir / "True.csv"

    archive_fake = Path(r"F:\archive\Fake.csv")
    archive_true = Path(r"F:\archive\True.csv")

    fake_path = workspace_fake if workspace_fake.exists() else archive_fake
    true_path = workspace_true if workspace_true.exists() else archive_true

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            "Dataset files not found. Place Fake.csv and True.csv in the project folder "
            "or ensure F:\\archive\\Fake.csv and F:\\archive\\True.csv exist."
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    return fake_df, true_df


ensure_nltk_resources()
fake, true = load_datasets()

# Add labels
fake["label"] = 0   # Fake news
true["label"] = 1   # True news

# Merge datasets
data = pd.concat([fake, true], axis=0)

# Shuffle dataset
data = data.sample(frac=1)

# Reset index
data.reset_index(drop=True, inplace=True)

# Save merged dataset
data.to_csv("news_dataset.csv", index=False)

print("Dataset merged successfully!")

# Check data
print(data.head())
#data analysis
data.info()
print(data['label'].value_counts())
print(data.shape)

# Check for missing values
print(data.isnull().sum())
#no missing values found so not using any imputation techniques

data.reset_index(inplace=True)
print(data.head())

#drop unnecessary columns
required_columns = {"title", "date", "text"}
missing_columns = required_columns.difference(data.columns)
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

data = data.drop(["date", "text"], axis=1)
print(data.head())

#preprocessing
#lemmatization
lm = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
corpus = []
for i in range(len(data)):
    review = re.sub("[^a-zA-Z]", " ", data["title"][i])
    review = review.lower()
    review = review.split()
    review = [lm.lemmatize(word) for word in review if word not in stop_words]
    review = " ".join(review)
    corpus.append(review)
print(len(corpus))
print(data['title'][0])

#vectorization
tf=TfidfVectorizer( max_features=5000, ngram_range=(1,3))
x = tf.fit_transform(corpus ).toarray()
print(x)

y=data['label']
print(y.head())


#data splitting
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=10,stratify=y)
print(x_train.shape[0], len(y_train))
print(x_test.shape[0], len(y_test))

#model training
random_forest = RandomForestClassifier()
rf=random_forest.fit(x_train, y_train)

#model evaluation
y_pred = rf.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
