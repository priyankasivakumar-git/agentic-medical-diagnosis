def preprocess_clinical(age, gender, fever, cough):
    """
    Converts raw clinical inputs into the 4-value array the model expects,
    in the order: [age, gender, fever, cough]

    Based on the real Phase 1 labels CSV:
    - age: used as-is (raw value, e.g. 54) — NOT divided by 100
    - gender: 0=female, 1=male
    - fever: 0=no, 1=yes
    - cough: 0=no, 1=yes
    """
    age_val = float(age)
    gender_val = 1 if str(gender).lower().startswith("m") or str(gender) == "1" else 0
    fever_val = 1 if fever else 0
    cough_val = 1 if cough else 0

    return [age_val, gender_val, fever_val, cough_val]