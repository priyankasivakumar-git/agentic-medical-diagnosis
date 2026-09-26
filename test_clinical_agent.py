from agents.clinical_agent import run_clinical_agent

result = run_clinical_agent(age=54, gender="male", fever=True, cough=True)
print(result)