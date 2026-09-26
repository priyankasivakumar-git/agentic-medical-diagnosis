import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import warnings
warnings.filterwarnings("ignore")

import shutil
import json
import time
from datetime import datetime

from agents.master_agent import run_pipeline

# ---- Edit this list: 2-3 varied patients you want as case studies ----
CASES = [
    {
        "name": "case_1",
        "image_path": "samples/patient1_xray.jpeg",
        "age": 54, "gender": "male", "fever": True, "cough": True,
    },
    {
        "name": "case_2",
        "image_path": "samples/c2.png",   # <-- change to a real second image
        "age": 34, "gender": "female", "fever": True, "cough": True,
    },
    {
        "name": "case_3",
        "image_path": "samples/T1.png",   # <-- change to a real third image
        "age": 75, "gender": "male", "fever": False, "cough": True,
    },
]

BASE_DIR = "outputs/paper_assets/case_studies"
os.makedirs(BASE_DIR, exist_ok=True)

for case in CASES:
    print(f"\n=== Running {case['name']} ===")

    if not os.path.exists(case["image_path"]):
        print(f"  SKIPPED — image not found: {case['image_path']}")
        continue

    case_dir = os.path.join(BASE_DIR, case["name"])
    os.makedirs(case_dir, exist_ok=True)

    try:
        report_text = run_pipeline(
            image_path=case["image_path"],
            age=case["age"], gender=case["gender"],
            fever=case["fever"], cough=case["cough"],
        )
    except Exception as e:
        print(f"  ERROR running pipeline: {e}")
        print("  (If this is a 429 rate-limit error, wait a bit and re-run just this case.)")
        continue

    # Save patient info
    with open(os.path.join(case_dir, "patient_info.txt"), "w") as f:
        f.write(f"Image: {case['image_path']}\n")
        f.write(f"Age: {case['age']}\n")
        f.write(f"Gender: {case['gender']}\n")
        f.write(f"Fever: {case['fever']}\n")
        f.write(f"Cough: {case['cough']}\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")

    # Save the Master Agent's full report
    with open(os.path.join(case_dir, "master_agent_report.md"), "w", encoding="utf-8") as f:
        f.write(report_text)

    # Copy the freshly generated Grad-CAM and SHAP images into this case's folder
    for src_name, dest_name in [("outputs/gradcam_result.png", "gradcam.png"),
                                  ("outputs/shap_result.png", "shap.png")]:
        if os.path.exists(src_name):
            shutil.copy(src_name, os.path.join(case_dir, dest_name))

    # Copy the original X-ray for reference
    shutil.copy(case["image_path"], os.path.join(case_dir, "input_xray" + os.path.splitext(case["image_path"])[1]))

    # Copy the newest log file (the one this run just created) into the case folder too
    logs_dir = "outputs/logs"
    if os.path.exists(logs_dir):
        newest_log = max(
            (os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".json")),
            key=os.path.getmtime, default=None
        )
        if newest_log:
            shutil.copy(newest_log, os.path.join(case_dir, "execution_log.json"))

    print(f"  Saved case package to: {case_dir}")
    time.sleep(2)  # small pause between cases, gentler on rate limits

print(f"\nDone. Case study packages saved under {BASE_DIR}/")