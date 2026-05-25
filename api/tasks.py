import numpy as np
from PIL import Image
import os
from .celery_app import celery
from inference.patching import enhance_large_image
from models.srresnet import build_srresnet

# Ініціалізуємо модель глобально
model = None

def get_model():
    global model
    if model is None:
        print("Loading model weights...")
        model = build_srresnet()
        model.load_weights("weights/generator_patchgan_epoch_19.weights.h5")
        print("Model loaded successfully!")
    return model

@celery.task(bind=True)
def process_image(self, file_path):
    self.update_state(state="STARTED")
    self.update_state(state="PROGRESS", meta={"progress": 1})

    def update_progress(p):
        #if p % 5 == 0:
        self.update_state(
            state="PROGRESS",
            meta={"progress": p}
        )

    sr_model = get_model()

    # Відкриваємо файл за шляхом, який передав API
    image = Image.open(file_path).convert("RGB")
    image = np.array(image).astype("float32") 

    sr = enhance_large_image(image, sr_model, progress_callback=update_progress)

    # Пост-обробка (обрізаємо значення та міняємо тип на картинку)
    sr = np.clip(sr, 0, 255).astype("uint8")
    img = Image.fromarray(sr)

    # Створюємо назву для результату на основі вхідного імені
    filename = os.path.basename(file_path)
    result_filename = f"enhanced_{filename.split('.')[0]}.png"
    result_path = os.path.join("/app/results", result_filename)
    
    img.save(result_path, format="PNG")
    
    return result_path