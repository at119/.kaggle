# preprocessing.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    #Dataset
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "creditcard.csv")
    df = pd.read_csv(csv_path)
    print("Loaded dataset:", df.shape)

    #samples for testing
    df = df.sample(frac=0.2, random_state=42)
    print("Using smaller sample:", df.shape)

    #Amount scale
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    #features and labels
    X = df.drop("Class", axis=1)
    y = df["Class"]

    #train sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train size:", X_train.shape, "| Test size:", X_test.shape)
    return X_train, X_test, y_train, y_test
