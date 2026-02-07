
import os
from sklearn.metrics import accuracy_score
from train_model import train_model

def evaluate():
    y_test, y_pred = train_model()

    
    accuracy = accuracy_score(y_test, y_pred)

    print("\n  Accuracy Result ")
    print(f"Accuracy: {accuracy:.4f}")

    
    log_folder = os.path.join(os.path.dirname(__file__), "../logs")
    os.makedirs(log_folder, exist_ok=True)

    with open(os.path.join(log_folder, "ml_results.txt"), "w", encoding="utf-8") as f:
        f.write("Supervised Learning - Credit Card Fraud Detection\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

    

if __name__ == "__main__":
    evaluate()
