# Appendix: Sample Multi-Agent Execution Log

*Source file:* run_20260901_195235.json  
*Timestamp:* 2026-09-01T19:52:35

This log records the full sequence of tool calls made by the Master Agent during one complete diagnostic run, demonstrating the system's audit trail and orchestration order. Long numeric feature arrays have been truncated for readability.

json
{
  "timestamp": "2026-09-01T19:52:35",
  "tool_calls": [
    {
      "agent": "image_analysis_agent",
      "output": {
        "feature_path": "outputs\\image_features.npy",
        "feature_vector_length": 1024,
        "image_importance_weight": 0.9907,
        "feature_summary": {
          "mean": 0.3745,
          "max": 6.0117,
          "nonzero_count": 804
        }
      }
    },
    {
      "agent": "clinical_analysis_agent",
      "output": {
        "feature_path": "outputs\\clinical_features.npy",
        "feature_vector_length": 64,
        "clinical_importance_weight": 0.0,
        "input_values_used": {
          "age": 75,
          "gender": "male",
          "fever": false,
          "cough": true
        }
      }
    },
    {
      "agent": "disease_prediction_agent",
      "output": {
        "predicted_class": "TUBERCULOSIS",
        "confidence_score": 1.0,
        "class_probabilities": {
          "COVID": 0.0,
          "Lung_Opacity": 0.0,
          "NORMAL": 0.0,
          "PNEUMONIA": 0.0,
          "TUBERCULOSIS": 1.0
        }
      }
    },
    {
      "agent": "explainability_agent",
      "output": {
        "gradcam_image_path": "outputs\\gradcam_result.png",
        "shap_image_path": "outputs\\shap_result.png",
        "shap_values": {
          "Age": 0.0,
          "Gender": 0.0,
          "Fever": 0.0,
          "Cough": 0.0
        },
        "explanation_summary": "Grad-CAM heatmap generated highlighting the image regions most influential to the prediction. SHAP values show each clinical feature's contribution toward the 'TUBERCULOSIS' prediction."
      }
    },
    {
      "agent": "final_report",
      "output": "# Multimodal Chest Disease Diagnosis Report\n\n## Patient Demographics & Clinical Information\n- **Age:** 75\n- **Gender:** Male\n- **Fever:** False\n- **Cough:** True\n- **X-Ray Image:** `samples/T1.png`\n\n---\n\n## Multimodal Analysis Results\n- **Predicted Disease:** TUBERCULOSIS\n- **Confidence Score:** 100.0% (1.0)\n- **Class Probabilities:**\n  - **TUBERCULOSIS:** 1.0000 (100.0%)\n  - **COVID:** 0.0000 (0.0%)\n  - **Lung_Opacity:** 0.0000 (0.0%)\n  - **NORMAL:** 0.0000 (0.0%)\n  - **PNEUMONIA:** 0.0000 (0.0%)\n\n---\n\n## Modality Importance Weights\n- **Image Importance Weight:** 0.9907 (99.07%)\n- **Clinical Importance Weight:** 0.0093 (0.93%)\n\n*Note on Modality Weights:* There is a very strong disagreement between the modalities, with the image analysis agent heavily dominating the decision-making process (99.07% weight) compared to the clinical data (0.93%). This indicates the model relied almost entirely on the visual features extracted from the chest X-ray.\n\n---\n\n## Explainability & Outputs\n- **Grad-CAM Heatmap Path:** `outputs\\gradcam_result.png`\n- **SHAP Values Plot Path:** `outputs\\shap_result.png`\n\n---\n\n## Clinician Plain-Language Explanation\nThe multimodal diagnostic pipeline evaluated both the patient's clinical inputs (age 75, male, no fever, cough present) and the chest X-ray image (`samples/T1.png`). The adaptive importance gate assigned an overwhelming 99.07% importance weight to the chest X-ray and 0.93% to the clinical data. Based on visual patterns extracted by DenseNet121, the model predicted **TUBERCULOSIS** with 100% confidence. \n\nThe accompanying Grad-CAM heatmap (`outputs\\gradcam_result.png`) highlights the specific regions of the lungs that heavily influenced this diagnosis."
    }
  ]
}

