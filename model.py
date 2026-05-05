import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils import clean_text

# ✅ MULTIPLE DATASETS
DATA_PATHS = [
    "archive/training/fakeNewsDataset",
    "archive/training/celebrityDataset"
]

print("Checking dataset paths...")
for path in DATA_PATHS:
    if not os.path.exists(path):
        raise Exception(f"❌ Path not found: {path}")
    print(f"✅ {path} → {os.listdir(path)}")


# 🔹 LOAD DATA FROM MULTIPLE DATASETS
def load_data(paths):
    data = []

    for folder_path in paths:
        for label_folder in ["fake", "legit"]:
            folder = os.path.join(folder_path, label_folder)

            label = 0 if label_folder == "fake" else 1

            for file_name in os.listdir(folder):
                file_path = os.path.join(folder, file_name)

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()

                        # 🔥 FILTER BAD/SHORT TEXT
                        if len(text.strip()) < 150:
                            continue

                        data.append([text, label])
                except:
                    continue

    return pd.DataFrame(data, columns=["text", "label"])


print("\n📂 Loading datasets...")
df = load_data(DATA_PATHS)

print(f"✅ Total samples after cleaning: {len(df)}")


# 🔹 PREPROCESS
print("\n🧹 Cleaning text...")
df["text"] = df["text"].apply(clean_text)


# 🔹 FEATURES
X = df["text"]
y = df["label"]


# 🔹 SPLIT (BALANCED)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 🔥 STRONG TF-IDF
vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),     # 🔥 BIG IMPROVEMENT
    max_df=0.9,
    min_df=2,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# 🔥 STRONG MODEL
model = LogisticRegression(
    max_iter=4000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)


# 🔹 EVALUATION
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔥 FINAL Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))


# 🔹 SAVE MODEL
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n💾 Model and vectorizer saved successfully!")