from agents.image_agent import run_image_agent
from agents.clinical_agent import run_clinical_agent
from agents.prediction_agent import run_prediction_agent

image_result = run_image_agent("samples/patient1_xray.jpeg")
clinical_result = run_clinical_agent(age=54, gender="male", fever=True, cough=True)

prediction = run_prediction_agent(
    image_feature_path=image_result["feature_path"],
    clinical_feature_path=clinical_result["feature_path"],
)

print("=== PREDICTION RESULT (via split agents) ===")
print(prediction)