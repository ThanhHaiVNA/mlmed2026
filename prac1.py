import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===== 1. Load dữ liệu (dùng path của bạn) =====
train_path = "/Users/thanhhai/Projects1/python/medicine/practice1/mitbih_train.csv"
test_path  = "/Users/thanhhai/Projects1/python/medicine/practice1/mitbih_test.csv"

train = pd.read_csv(train_path, header=None)
test  = pd.read_csv(test_path, header=None)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# ===== 2. Tách X, y =====
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

X_test  = test.iloc[:, :-1].values
y_test  = test.iloc[:, -1].values

# ===== 3. Build Random Forest =====
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ===== 4. Train =====
print("Training Random Forest...")
rf.fit(X_train, y_train)

# ===== 5. Test =====
print("Testing...")
y_pred = rf.predict(X_test)

# ===== 6. Evaluation =====
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
