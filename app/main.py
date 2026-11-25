import io, json, os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
LABELS = (ROOT / "model/labels.txt").read_text().split()
METRICS = json.loads((ROOT / "model/metrics.json").read_text())

MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "model/best_final.keras"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH

if MODEL_PATH.suffix == ".keras":
    import tensorflow as tf
    _model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    def run(x):
        return _model(x, training=False).numpy()[0]
else:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
    _interp = Interpreter(model_path=str(MODEL_PATH), num_threads=2)
    _interp.allocate_tensors()
    _in, _out = _interp.get_input_details()[0], _interp.get_output_details()[0]

    def run(x):
        _interp.set_tensor(_in["index"], x)
        _interp.invoke()
        return _interp.get_tensor(_out["index"])[0]

print(f"[cyberleaf] model: {MODEL_PATH.name}")

app = FastAPI(title="CyberLeaf // Crop Disease Analysis")

@app.get("/api/meta")
def meta():
    return {"labels": LABELS, "model": MODEL_PATH.name, **METRICS}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        img = ImageOps.exif_transpose(img).convert("RGB").resize((224, 224), Image.BILINEAR)
    except Exception:
        raise HTTPException(400, "Unreadable image")
    x = np.asarray(img, dtype=np.float32)[None]  # raw 0-255, normalisation is inside the model
    p = run(x)
    order = np.argsort(-p)
    return JSONResponse({"top": [{"label": LABELS[i], "prob": float(p[i])} for i in order]})

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")