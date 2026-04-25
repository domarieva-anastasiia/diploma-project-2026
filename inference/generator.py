import tensorflow as tf
from models.srresnet import build_srresnet

class SRGANGenerator:
    def __init__(self, weights_path):
        self.model = build_srresnet()
        self.model.load_weights(weights_path)

    def enhance(self, image):
        image = tf.expand_dims(image, axis=0)  # [H,W,3] → [1,H,W,3]
        sr = self.model(image, training=False)
        sr = tf.squeeze(sr, axis=0)
        return sr