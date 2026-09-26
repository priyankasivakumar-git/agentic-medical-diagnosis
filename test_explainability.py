from agents.image_agent import run_image_agent
from agents.explainability_agent import run_explainability_agent

image_result = run_image_agent("samples/patient1_xray.jpeg")

result = run_explainability_agent(
    image_path="samples/patient1_xray.jpeg",
    image_feature_path=image_result["feature_path"],
    age=54,
    gender="male",
    fever=True,
    cough=True,
)

print(result)