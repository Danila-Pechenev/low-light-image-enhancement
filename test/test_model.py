from PIL import Image
import io
import os
from ..app import model

model_instance = model.create_model()


def template(filename: str):
    image = Image.open(filename)
    width, height = image.size
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    output_image = model.run_model(image_bytes, model_instance)
    return width, height, output_image


def test_image_jpg():
    width, height, output_image = template(os.path.join(os.getcwd(), "test/test_images/test1.jpg"))

    assert width == output_image.size[0]
    assert height == output_image.size[1]


def test_image_png():
    width, height, output_image = template(os.path.join(os.getcwd(), "test/test_images/test2.png"))

    assert width == output_image.size[0]
    assert height == output_image.size[1]


def test_image_jpeg():
    width, height, output_image = template(os.path.join(os.getcwd(), "test/test_images/test3.jpeg"))

    assert width == output_image.size[0]
    assert height == output_image.size[1]
