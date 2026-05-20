from flask import Flask, request, send_file, send_from_directory
import numpy as np
from PIL import Image
import io
import uuid
import os
import tensorflow as tf

from inference.generator import SRGANGenerator
from inference.patching import enhance_large_image
from api.tasks import process_image
from celery.result import AsyncResult
from api.celery_app import celery

app = Flask(__name__)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_file(file):
    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    file.save(file_path)

    return file_path

@app.route("/")
def home():
    return "API is running"

# @app.route("/enhance", methods=["POST"])
# def enhance():
#     file = request.files["image"]

#     image = Image.open(file).convert("RGB")
#     image = np.array(image).astype("float32")

#     sr = enhance_large_image(image, model.model)

#     sr = np.clip(sr, 0, 255).astype("uint8")

#     img = Image.fromarray(sr)

#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     buf.seek(0)

#     return send_file(buf, mimetype="image/png")



@app.route("/enhance", methods=["POST"])
def enhance():
    file = request.files["image"]

    file_path = os.path.join("/app/uploads", file.filename)
    file.save(file_path)

    task = process_image.delay(file_path)

    return {"job_id": task.id}


@app.route("/result/<job_id>")
def result(job_id):
    task = AsyncResult(job_id)

    file_path = task.result  
    try:
        return send_file(file_path, mimetype='image/png')
    except Exception as e:
        return {"status": "error", "message": e}, 404
    

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory("/app/results", filename, mimetype='image/png')


@app.route("/status/<job_id>")
def status(job_id):
    task = AsyncResult(job_id)
    state = task.state

    if state == 'PENDING':
        return {"status": "PENDING", "progress": 0}, 202
    
    elif state == 'STARTED':
        return {"status": "STARTED", "progress": 0}, 202
    
    elif state == 'PROGRESS':
        progress_val = task.info.get("progress", 0) if isinstance(task.info, dict) else 0
        return {"status": "PROGRESS", "progress": progress_val}, 202
    
    elif task.state == 'FAILURE':
        return {"status": "failed", "error": str(task.info)}, 500
    
    elif task.state == 'SUCCESS':
        # task.result містить шлях, який повернула функція enhance_large_image
        file_path = task.result
         
        filename = os.path.basename(file_path)
        return {
            "status": "SUCCESS", 
            "progress": 100, 
            "download_url": f"/download/{filename}"
        }, 200

    return {"status": state}, 202


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    celery.control.revoke(job_id, terminate=True)
    return {"status": "cancelled"}



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)