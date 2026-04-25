from flask import Flask, request, send_file
import numpy as np
from PIL import Image
import io

from inference.generator import SRGANGenerator
from inference.patching import enhance_large_image

app = Flask(__name__)

model = SRGANGenerator("weights/srresnet_finetuned_v6.weights.h5")

@app.route("/")
def home():
    return "API is running"

@app.route("/enhance", methods=["POST"])
def enhance():
    file = request.files["image"]

    image = Image.open(file).convert("RGB")
    image = np.array(image).astype("float32")

    sr = enhance_large_image(image, model.model)

    sr = np.clip(sr, 0, 255).astype("uint8")

    img = Image.fromarray(sr)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)