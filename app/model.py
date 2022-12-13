from tensorflow import keras
from PIL import Image
import numpy as np
import io


def create_model() -> keras.Model:
    return keras.models.load_model("model")


def run_model(image: io.BytesIO, model: keras.Model) -> Image.Image:
    image: Image.Image = Image.open(image)
    width: int = image.size[0]
    height: int = image.size[1]
    image: Image.Image = image.resize((960, 640))
    image: np.ndarray = keras.utils.img_to_array(image)
    image: np.ndarray = image.astype("float32") / 255.0
    image: np.ndarray = np.expand_dims(image, axis=0)
    output: np.ndarray = model.predict(image)
    output_image: np.ndarray = output[0] * 255.0
    output_image: np.ndarray = output_image.clip(0, 255)
    output_image: np.ndarray = output_image.reshape(
        (np.shape(output_image)[0], np.shape(output_image)[1], 3)
    )
    output_image: np.ndarray = np.uint32(output_image)
    output_image: np.ndarray = output_image.astype(np.uint8)
    output_image: Image.Image = Image.fromarray(output_image)
    output_image: Image.Image = output_image.resize((width, height))
    output_image.save("user_data/output.jpg")
    return output_image
