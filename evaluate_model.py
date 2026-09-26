import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from agents.image_agent import run_image_agent
from agents.clinical_agent import run_clinical_agent
from agents.prediction_agent import run_prediction_agent

CLASS_NAMES = ["COVID", "Lung_Opacity", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

LABELS_CSV = "dataset_encoded.csv"   # <-- change this to your actual CSV filename/path

df = pd.read_csv(LABELS_CSV)
print(f"Loaded {len(df)} rows from {LABELS_CSV}")
print("Label distribution in CSV:")
print(df["label"].value_counts().sort_index())
print()

y_true = []
y_pred = []
errors = []

for i, row in df.iterrows():
    image_path = row["image_path"]
    if not os.path.exists(image_path):
        errors.append((image_path, "file not found"))
        continue

    try:
        image_result = run_image_agent(image_path)
        clinical_result = run_clinical_agent(
            age=row["age"], gender=row["gender"], fever=bool(row["fever"]), cough=bool(row["cough"])
        )
        prediction = run_prediction_agent(
            image_feature_path=image_result["feature_path"],
            clinical_feature_path=clinical_result["feature_path"],
        )
        pred_index = CLASS_NAMES.index(prediction["predicted_class"])

        y_true.append(int(row["label"]))
        y_pred.append(pred_index)

    except Exception as e:
        errors.append((image_path, str(e)))

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(df)}...")

print(f"\nDone. {len(y_true)} evaluated successfully, {len(errors)} errors.")
if errors:
    print("First few errors:", errors[:5])

# --- Metrics ---
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)

print("\n=== CONFUSION MATRIX ===")
print("Rows = actual, Columns = predicted")
print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

print(f"\nOverall Accuracy: {acc:.4f}")
print("\n=== PER-CLASS METRICS ===")
for cls in CLASS_NAMES:
    m = report[cls]
    print(f"{cls:15s} precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1-score']:.3f}  support={int(m['support'])}")

# --- Save results for the dashboard ---
os.makedirs("outputs", exist_ok=True)
results = {
    "class_names": CLASS_NAMES,
    "confusion_matrix": cm.tolist(),
    "accuracy": acc,
    "classification_report": report,
}
with open("outputs/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved results to outputs/evaluation_results.json")