import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

from agents.image_agent import run_image_agent
from agents.clinical_agent import run_clinical_agent
from agents.prediction_agent import run_prediction_agent
from agents.explainability_agent import run_explainability_agent
from memory_store import Memory

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_current_memory = None


def image_analysis_agent(image_path: str) -> dict:
    """Extracts weighted visual features from a chest X-ray using DenseNet121
    and an adaptive importance gate. Returns a path to the saved feature file
    and a summary. Call this FIRST, before prediction.

    Args:
        image_path: File path to the chest X-ray image.
    """
    result = run_image_agent(image_path)
    _current_memory.log("image_analysis_agent", result)
    return result


def clinical_analysis_agent(age: float, gender: str, fever: bool, cough: bool) -> dict:
    """Extracts weighted clinical features from patient data (age, gender, fever,
    cough) using a dense network and an adaptive importance gate. Returns a path
    to the saved feature file. Call this FIRST, before prediction, alongside
    image_analysis_agent.

    Args:
        age: Patient age in years.
        gender: Patient gender, e.g. 'male' or 'female'.
        fever: Whether the patient has a fever.
        cough: Whether the patient has a cough.
    """
    result = run_clinical_agent(age, gender, fever, cough)
    _current_memory.log("clinical_analysis_agent", result)
    return result


def disease_prediction_agent(image_feature_path: str, clinical_feature_path: str) -> dict:
    """Fuses image features and clinical features to predict the disease class,
    confidence score, and probabilities for all 5 classes. Call this AFTER both
    image_analysis_agent and clinical_analysis_agent have run.

    Args:
        image_feature_path: File path to the saved image feature .npy file,
            from image_analysis_agent's output.
        clinical_feature_path: File path to the saved clinical feature .npy file,
            from clinical_analysis_agent's output.
    """
    result = run_prediction_agent(image_feature_path, clinical_feature_path)
    _current_memory.log("disease_prediction_agent", result)
    return result


def explainability_agent(image_path: str, image_feature_path: str, age: float, gender: str, fever: bool, cough: bool) -> dict:
    """Generates a Grad-CAM heatmap (image explanation) and SHAP values (clinical
    feature importance) explaining the prediction. Call this LAST, after
    disease_prediction_agent and using the image_feature_path from
    image_analysis_agent's earlier output.

    Args:
        image_path: File path to the original chest X-ray image.
        image_feature_path: File path to the saved image feature .npy file
            from image_analysis_agent's output.
        age: Patient age in years.
        gender: Patient gender.
        fever: Whether the patient has a fever.
        cough: Whether the patient has a cough.
    """
    result = run_explainability_agent(image_path, image_feature_path, age, gender, fever, cough)
    _current_memory.log("explainability_agent", result)
    return result


SYSTEM_PROMPT = """You are the Master Agent coordinating a multimodal chest disease
diagnosis pipeline with four specialist agents.

Your job, in this exact order:
1. Call image_analysis_agent on the X-ray image.
2. Call clinical_analysis_agent on the patient's clinical data.
3. Call disease_prediction_agent, passing the feature file paths returned by
   steps 1 and 2.
4. Call explainability_agent to generate visual evidence for the prediction,
   passing the image_feature_path from step 1's output.
5. Produce a final decision-support report including: predicted disease,
   confidence score, class probabilities, each agent's importance weight
   (image vs clinical), where the Grad-CAM image was saved, and a short
   plain-language explanation a clinician could read.

Always call the tools to get real data — never invent probabilities, weights,
or file paths. If the image and clinical importance weights disagree strongly
(e.g. one very high, one very low), note this as a point of interest in your
final report.
"""


def run_pipeline(image_path: str, age: float, gender: str, fever: bool, cough: bool) -> str:
    global _current_memory
    _current_memory = Memory()

    chat = client.chats.create(
        model="gemini-3.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[
                image_analysis_agent,
                clinical_analysis_agent,
                disease_prediction_agent,
                explainability_agent,
            ],
        ),
    )

    prompt = (
        f"Diagnose this patient. Image path: {image_path}. "
        f"Age: {age}, Gender: {gender}, Fever: {fever}, Cough: {cough}."
    )

    response = chat.send_message(prompt)

    _current_memory.log("final_report", response.text)
    log_path = _current_memory.save()
    print(f"(Run log saved to: {log_path})")

    return response.text