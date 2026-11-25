# CYBERLEAF // Crop Disease Analysis
Local:   pip install -r requirements.txt && uvicorn app.main:app --reload   ->  http://localhost:8000
Docker:  docker build -t cyberleaf . && docker run -p 8000:8000 cyberleaf
Swap model: replace model/model.tflite (e.g. model_dynamic_range.tflite for a smaller build).
