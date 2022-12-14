from huggingface_hub import from_pretrained_keras
from tensorflow import keras
from PIL import Image
import numpy as np
import io
import os


def create_model() -> keras.Model:
    return from_pretrained_keras("keras-io/lowlight-enhance-mirnet")


def run_model(image_bytes: io.BytesIO, model: keras.Model) -> Image.Image:
    image = Image.open(image_bytes)
    width, height = image.size
    image = image.resize((960, 640))
    image_array = keras.utils.img_to_array(image)
    image_array = image_array.astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    output = model.predict(image_array)
    output_image_array = output[0] * 255.0
    output_image_array = output_image_array.clip(0, 255)
    output_image_array = output_image_array.reshape(
        (np.shape(output_image_array)[0], np.shape(output_image_array)[1], 3)
    )
    output_image_array = output_image_array.astype(np.uint8)
    output_image = Image.fromarray(output_image_array)
    output_image = output_image.resize((width, height))
    if not os.path.exists("user_data"):
        os.makedirs("user_data")
    output_image.save("user_data/output.jpg")
    return output_image
