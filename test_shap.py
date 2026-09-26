from agents.image_agent import run_image_agent
from agents.explainability_agent import run_shap_explanation

# Reuse an already-computed image feature file (run test_image_agent.py first if needed)
image_result = run_image_agent("samples/patient1_xray.jpeg")

result = run_shap_explanation(
    image_feature_path=image_result["feature_path"],
    age=54,
    gender="male",
    fever=True,
    cough=True,
)

print(result)