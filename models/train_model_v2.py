#!/usr/bin/env python3
"""
STEP 5: Feature Engineering — Raw Prices Ki Jagah Ratios/Returns
--------------------------------------------------------------------
Same model (Logistic Regression) rakha hai, SIRF features badle hain.
Isse pata chalega ki improvement features se aaya ya model se.

Chalane ka tarika:
    python3 train_model_v2.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Support both running from `models/` directory or root workspace
if os.path.exists("ml_dataset.csv"):
    CSV_FILE = "ml_dataset.csv"
elif os.path.exists("../ml_dataset.csv"):
    CSV_FILE = "../ml_dataset.csv"
else:
    CSV_FILE = "/Users/adarshvashistha/Desktop/EZ/ml_dataset.csv"


def engineer_features(df):
    """
    Raw prices ki jagah stock-agnostic features banata hai.
    IMPORTANT: Ye per-symbol group karke karna hai, warna ek stock ka
    data doosre stock ke saath mix ho jayega (jaise lag/diff calculation mein).
    """
    df = df.sort_values(["Symbol", "Date"]).copy()

    engineered_rows = []

    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values("Date").copy()

        # 1. Daily Return % — aaj ka open se close tak kitna % change hua
        group["Daily_Return_Pct"] = ((group["Close"] - group["Open"]) / group["Open"]) * 100

        # 2. Intraday Spread % — din mein kitna volatility tha
        group["Intraday_Spread_Pct"] = ((group["High"] - group["Low"]) / group["Open"]) * 100

        # 3. Volume Change % — kal ke comparison mein aaj volume kitna badla
        group["Volume_Change_Pct"] = group["Volume"].pct_change() * 100

        # 4. Sentiment Lag — KAL ka sentiment (aaj ka nahi, taaki "future leak" na ho)
        #    Ye important hai: aaj ke price ko predict karne ke liye kal ka sentiment use karo
        group["Sentiment_Lag_1"] = group["AvgSentiment"].shift(1)

        # 5. Price momentum — pichhle 3 din ka average return (simple moving trend)
        group["Return_MA_3"] = group["Daily_Return_Pct"].rolling(window=3).mean()

        engineered_rows.append(group)

    result = pd.concat(engineered_rows, ignore_index=True)

    # Pehle kuch rows mein NaN aayenge (rolling/shift ki wajah se) — unhe hatao
    before = len(result)
    result = result.dropna(subset=[
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct",
        "Sentiment_Lag_1", "Return_MA_3", "NextDayDirection"
    ])
    result["NextDayDirection"] = result["NextDayDirection"].astype(int)
    print(f"Rows before feature engineering cleanup: {before}, after: {len(result)}")

    return result


def split_train_test_by_time(df, test_fraction=0.2):
    df = df.sort_values("Date")
    split_index = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_index], df.iloc[split_index:]


def get_baseline_accuracy(train_df, test_df):
    most_common_class = train_df["NextDayDirection"].mode()[0]
    baseline_predictions = [most_common_class] * len(test_df)
    label_name = "UP (1)" if most_common_class == 1 else "DOWN (0)"
    return accuracy_score(test_df["NextDayDirection"], baseline_predictions), label_name


def train_model(train_df, test_df, feature_columns):
    X_train, y_train = train_df[feature_columns], train_df["NextDayDirection"]
    X_test, y_test = test_df[feature_columns], test_df["NextDayDirection"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy, predictions, y_test


def main():
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=["NextDayDirection"])

    print("Engineering features (returns, ratios, lag)...")
    df = engineer_features(df)

    # NAYA feature set — raw prices HATA diye, sirf ratios/returns rakhe
    feature_columns = [
        "Daily_Return_Pct", "Intraday_Spread_Pct", "Volume_Change_Pct",
        "Sentiment_Lag_1", "Return_MA_3", "NewsCount"
    ]

    train_df, test_df = split_train_test_by_time(df)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Features used: {feature_columns}\n")

    baseline_acc, common_class = get_baseline_accuracy(train_df, test_df)
    print(f"BASELINE (hamesha predict '{common_class}'): {baseline_acc*100:.1f}% accuracy")

    model, model_acc, predictions, y_test = train_model(train_df, test_df, feature_columns)
    print(f"LOGISTIC REGRESSION (with engineered features): {model_acc*100:.1f}% accuracy")

    print("\n" + "="*60)
    if model_acc > baseline_acc:
        print(f"✅ Model baseline se {(model_acc-baseline_acc)*100:.1f} points BETTER hai!")
        print("   -> Feature engineering ne kaam kiya. Ab Random Forest try karo.")
    else:
        print(f"❌ Abhi bhi baseline se neeche/barabar hai ({(model_acc-baseline_acc)*100:.1f} points).")
        print("   -> Matlab problem sirf raw-price nahi thi. Model type ya data quantity issue ho sakta hai.")
    print("="*60)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, labels=[0, 1], target_names=["DOWN", "UP"], zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("(Rows = actual, Columns = predicted. Format: [[TN, FP], [FN, TP]])")

    print("\nFeature Importance:")
    for feature, coef in zip(feature_columns, model.coef_[0]):
        print(f"  {feature:<20} {coef:+.3f}")


if __name__ == "__main__":
    main()
