from huggingface_hub import from_pretrained_keras
from tensorflow import keras
from PIL import Image
import numpy as np
import io


def create_model() -> keras.Model:
    return from_pretrained_keras("keras-io/lowlight-enhance-mirnet")


def run_model(image_bytes: io.BytesIO, model: keras.Model) -> Image.Image:
    image: Image.Image = Image.open(image_bytes)
    width: int
    height: int
    width, height = image.size
    image: Image.Image = image.resize((960, 640))
    image_array1: np.ndarray = keras.utils.img_to_array(image)
    image_array2: np.ndarray = image_array1.astype("float32") / 255.0
    image_array3: np.ndarray = np.expand_dims(image_array2, axis=0)
    output: np.ndarray = model.predict(image_array3)
    output_image_array1: np.ndarray = output[0] * 255.0
    output_image_array2: np.ndarray = output_image_array1.clip(0, 255)
    output_image_array3: np.ndarray = output_image_array2.reshape(
        (np.shape(output_image_array2)[0], np.shape(output_image_array2)[1], 3)
    )
    output_image_array4: np.ndarray = np.uint32(output_image_array3)
    output_image_array5: np.ndarray = output_image_array4.astype(np.uint8)
    output_image1: Image.Image = Image.fromarray(output_image_array5)
    output_image2: Image.Image = output_image1.resize((width, height))
    # Uncomment if necessary:
    # output_image.save("user_data/output.jpg")
    return output_image2
