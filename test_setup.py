import torch
import torchvision
import shap
import cv2
import os
from dotenv import load_dotenv

print("torch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
print("shap imported OK")
print("opencv imported OK")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print("API key loaded, starts with:", api_key[:12], "...")
else:
    print("WARNING: API key not found. Check your .env file.")