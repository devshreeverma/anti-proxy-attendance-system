import os
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from extract_features import extract_features


print("🔄 Extracting features...")

# ================= LOAD DATA =================
X, y, label_map = extract_features()

if len(X) == 0:
    print("❌ No data found! Check dataset.")
    exit()

print("✅ Data loaded:", X.shape)
print("📊 Label distribution:", Counter(y))


# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # ensures equal class distribution
)


# ================= SCALING =================
print("⚙️ Scaling features...")
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ================= MODEL + TUNING =================
print("🤖 Training SVM with hyperparameter tuning...")

param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']   # best default for your case
}

grid = GridSearchCV(
    SVC(),
    param_grid,
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Best model
model = grid.best_estimator_

print("🔥 Best Parameters Found:", grid.best_params_)


# ================= EVALUATION =================
print("\n📊 Evaluating model...")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = model.score(X_train, y_train)

print("🎯 Test Accuracy:", test_accuracy)
print("📈 Train Accuracy:", train_accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("📊 Confusion Matrix:\n", cm)

# Classification Report (VERY USEFUL)
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred))


# ================= SAVE MODEL =================
print("\n💾 Saving model...")

os.makedirs("../model", exist_ok=True)

with open("../model/voice_model.pkl", "wb") as f:
    pickle.dump((model, scaler, label_map), f)

print("✅ Model + scaler + label_map saved successfully!")