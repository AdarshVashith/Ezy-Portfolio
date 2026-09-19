#!/usr/bin/env python3
"""
STEP 4: First ML Model — Baseline vs Real Model
--------------------------------------------------
Ye script:
1. ml_dataset.csv load karta hai
2. Train/Test split karta hai (time-based, taaki future data leak na ho)
3. Ek BASELINE banata hai ("hamesha most common class predict karo")
4. Ek REAL model (Logistic Regression) train karta hai
5. Dono ko compare karta hai — asli test yehi hai, accuracy number nahi

Chalane ka tarika:
    python3 train_model.py
"""

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# ---------- CONFIG ----------
# Support both running from `models/` directory or root workspace
if os.path.exists("ml_dataset.csv"):
    CSV_FILE = "ml_dataset.csv"
elif os.path.exists("../ml_dataset.csv"):
    CSV_FILE = "../ml_dataset.csv"
else:
    CSV_FILE = "/Users/adarshvashistha/Desktop/EZ/ml_dataset.csv"


def load_and_prepare_data():
    """CSV load karke, missing/pending rows clean karta hai"""
    df = pd.read_csv(CSV_FILE)

    print(f"Total rows loaded: {len(df)}")

    # Wo rows hatao jinke liye target nahi hai (last day per symbol - "Pending")
    df = df.dropna(subset=["NextDayDirection"])
    df["NextDayDirection"] = df["NextDayDirection"].astype(int)
    print(f"Rows after removing 'pending' targets: {len(df)}")

    return df


def split_train_test_by_time(df, test_fraction=0.2):
    """
    IMPORTANT: Random split nahi karenge — time-series data mein
    'future' training set mein leak nahi honi chahiye.
    Isliye date ke hisaab se sort karke, aakhri X% ko test set banayenge.
    """
    df = df.sort_values("Date")

    split_index = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    return train_df, test_df


def get_baseline_accuracy(train_df, test_df):
    """
    Baseline: Sabse common class ko hamesha predict karo.
    Agar tumhara model isse better nahi hai, to wo kuch seekh hi nahi raha.
    """
    most_common_class = train_df["NextDayDirection"].mode()[0]
    baseline_predictions = [most_common_class] * len(test_df)

    baseline_acc = accuracy_score(test_df["NextDayDirection"], baseline_predictions)
    label_name = "UP (1)" if most_common_class == 1 else "DOWN (0)"
    return baseline_acc, label_name


def train_real_model(train_df, test_df, feature_columns):
    """Logistic Regression model train karta hai aur test set pe evaluate karta hai"""

    X_train = train_df[feature_columns]
    y_train = train_df["NextDayDirection"]
    X_test = test_df[feature_columns]
    y_test = test_df["NextDayDirection"]

    # Features ko scale karna zaroori hai (Volume jaisa bada number, Sentiment jaisa chhota number)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, predictions, y_test


def main():
    df = load_and_prepare_data()

    if len(df) < 20:
        print("\n⚠️  WARNING: Bahut kam data hai (< 20 rows) reliable training ke liye.")
        print("Result sirf 'pipeline test' ke liye hai, actual accuracy pe bharosa mat karo.\n")

    train_df, test_df = split_train_test_by_time(df)

    # Features jo model ko dikhayenge (ye tumhare column names hain)
    feature_columns = ["Open", "High", "Low", "Close", "Volume", "AvgSentiment", "NewsCount"]

    print(f"\nFeatures used: {feature_columns}\n")

    # Step 1: Baseline
    baseline_acc, common_class = get_baseline_accuracy(train_df, test_df)
    print(f"BASELINE (hamesha predict '{common_class}'): {baseline_acc*100:.1f}% accuracy")

    # Step 2: Real model
    model, model_acc, predictions, y_test = train_real_model(train_df, test_df, feature_columns)
    print(f"LOGISTIC REGRESSION MODEL:                {model_acc*100:.1f}% accuracy")

    # Step 3: Compare
    print("\n" + "="*50)
    if model_acc > baseline_acc:
        print(f"✅ Model baseline se {(model_acc-baseline_acc)*100:.1f} points BETTER hai — kuch seekh raha hai!")
    elif model_acc == baseline_acc:
        print("⚠️  Model baseline ke BARABAR hai — abhi kuch extra nahi seekh raha.")
    else:
        print("❌ Model baseline se BADTAR hai — kuch galat hai (overfitting ya bahut kam data).")
    print("="*50)

    # Detailed report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions, labels=[0, 1], target_names=["DOWN (0)", "UP (1)"], zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("(Rows = actual, Columns = predicted. Format: [[TN, FP], [FN, TP]])")

    # Feature importance (Logistic Regression coefficients)
    print("\nFeature Importance (higher absolute value = more influence):")
    for feature, coef in zip(feature_columns, model.coef_[0]):
        print(f"  {feature:<15} {coef:+.3f}")


if __name__ == "__main__":
    main()
