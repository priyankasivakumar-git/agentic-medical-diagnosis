import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

from agents.master_agent import run_pipeline

if __name__ == "__main__":
    result = run_pipeline(
        image_path="samples/patient1_xray.jpeg",
        age=54,
        gender="male",
        fever=True,
        cough=True,
    )
    print("=== FINAL DECISION SUPPORT ===")
    print(result)