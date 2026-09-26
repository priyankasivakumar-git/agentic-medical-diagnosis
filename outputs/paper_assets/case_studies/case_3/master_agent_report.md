# Multimodal Chest Disease Diagnosis Report

## Patient Demographics & Clinical Information
- **Age:** 75
- **Gender:** Male
- **Fever:** False
- **Cough:** True
- **X-Ray Image:** `samples/T1.png`

---

## Multimodal Analysis Results
- **Predicted Disease:** TUBERCULOSIS
- **Confidence Score:** 100.0% (1.0)
- **Class Probabilities:**
  - **TUBERCULOSIS:** 1.0000 (100.0%)
  - **COVID:** 0.0000 (0.0%)
  - **Lung_Opacity:** 0.0000 (0.0%)
  - **NORMAL:** 0.0000 (0.0%)
  - **PNEUMONIA:** 0.0000 (0.0%)

---

## Modality Importance Weights
- **Image Importance Weight:** 0.9907 (99.07%)
- **Clinical Importance Weight:** 0.0093 (0.93%)

*Note on Modality Weights:* There is a very strong disagreement between the modalities, with the image analysis agent heavily dominating the decision-making process (99.07% weight) compared to the clinical data (0.93%). This indicates the model relied almost entirely on the visual features extracted from the chest X-ray.

---

## Explainability & Outputs
- **Grad-CAM Heatmap Path:** `outputs\gradcam_result.png`
- **SHAP Values Plot Path:** `outputs\shap_result.png`

---

## Clinician Plain-Language Explanation
The multimodal diagnostic pipeline evaluated both the patient's clinical inputs (age 75, male, no fever, cough present) and the chest X-ray image (`samples/T1.png`). The adaptive importance gate assigned an overwhelming 99.07% importance weight to the chest X-ray and 0.93% to the clinical data. Based on visual patterns extracted by DenseNet121, the model predicted **TUBERCULOSIS** with 100% confidence. 

The accompanying Grad-CAM heatmap (`outputs\gradcam_result.png`) highlights the specific regions of the lungs that heavily influenced this diagnosis.