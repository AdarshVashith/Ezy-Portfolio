#!/usr/bin/env python3
"""
STEP 6: Non-Linear Model Test — Random Forest
--------------------------------------------------
Same engineered features (v2 wale), sirf model badla hai.
Isse pata chalega: kya problem "linear model" thi, ya signal khud weak hai.

Chalane ka tarika:
    python3 train_model_v3.py
"""

import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure models directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Support both running from `models/` directory or root workspace
if os.path.exists("ml_dataset.csv"):
    CSV_FILE = "ml_dataset.csv"
elif os.path.exists("../ml_dataset.csv"):
    CSV_FILE = "../ml_dataset.csv"
else:
    CSV_FILE = "/Users/adarshvashistha/Desktop/EZ/ml_dataset.csv"

# Reuse the same functions from v2
from train_model_v2 import engineer_features, split_train_test_by_time, get_baseline_accuracy


def train_random_forest(train_df, test_df, feature_columns):
    X_train, y_train = train_df[feature_columns], train_df["NextDayDirection"]
    X_test, y_test = test_df[feature_columns], test_df["NextDayDirection"]

    # Random Forest ko scaling ki zaroorat nahi hoti (tree-based model hai)
    model = RandomForestClassifier(
        n_estimators=100,     # 100 chhote decision trees banayega
        max_depth=5,          # zyada gehra na jaaye, warna overfit karega itne kam data pe
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy, predictions, y_test


def main():
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=["NextDayDirection"])

    print("Engineering features...")
    df = engineer_features(df)

    feature_columns = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct",
        "Sentiment_Lag_1", "Return_MA_3", "NewsCount"
    ]

    train_df, test_df = split_train_test_by_time(df)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}\n")

    baseline_acc, common_class = get_baseline_accuracy(train_df, test_df)
    print(f"BASELINE:                              {baseline_acc*100:.1f}%")

    model, model_acc, predictions, y_test = train_random_forest(train_df, test_df, feature_columns)
    print(f"RANDOM FOREST (same features as v2):   {model_acc*100:.1f}%")

    print("\n" + "="*60)
    diff = (model_acc - baseline_acc) * 100
    if diff > 1:
        print(f"✅ Random Forest baseline se {diff:.1f} points BETTER — non-linear signal mila!")
    elif diff > -1:
        print(f"⚠️  Baseline ke lagbhag barabar ({diff:+.1f} points) — model type se farq nahi pada.")
        print("   -> Iska matlab: signal khud weak hai, sirf 'linear vs non-linear' ka issue nahi tha.")
    else:
        print(f"❌ Random Forest bhi baseline se neeche hai ({diff:.1f} points).")
        print("   -> Strong signal jo bhi hai, ye features usse capture nahi kar rahe.")
    print("="*60)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, labels=[0, 1], target_names=["DOWN", "UP"], zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("(Rows = actual, Columns = predicted. Format: [[TN, FP], [FN, TP]])")

    print("\nFeature Importance (Random Forest ka apna measure):")
    importances = sorted(zip(feature_columns, model.feature_importances_), key=lambda x: -x[1])
    for feature, imp in importances:
        print(f"  {feature:<20} {imp:.3f}")


if __name__ == "__main__":
    main()
