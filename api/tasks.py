from .celery_app import celery
from inference.patching import enhance_large_image

@celery.task(bind=True)
def process_image(self, file_path):
    
    result_path = enhance_large_image(file_path)
    return result_path