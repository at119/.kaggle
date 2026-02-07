#trains and saves model predictions
from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_and_preprocess

def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print("Training model...")

    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")
    y_pred = model.predict(X_test)
    return y_test, y_pred
