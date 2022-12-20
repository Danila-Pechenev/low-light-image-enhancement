import telebot
from PIL import Image
import sys
import os
import io

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
import model

bot = telebot.TeleBot("<Paste your token here>")
model_instance = model.create_model()


@bot.message_handler(content_types=["photo"])
def photo(message):
    user_id = message.from_user.id
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    if not os.path.exists("user_data"):
        os.makedirs("user_data")
    filename = f"user_data/image_{user_id}_{file_id}.jpg"
    with open(filename, "wb") as new_file:
        new_file.write(downloaded_file)

    bot.send_message(message.from_user.id, "Изображение обрабатывается...")

    image = Image.open(filename)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    output_image = model.run_model(image_bytes, model_instance)
    os.remove(filename)

    bot.send_photo(user_id, output_image)


@bot.message_handler(content_types=["text"])
def start(message):
    bot.send_message(
        message.from_user.id,
        """Этот бот осветляет фотографии, сделанные при плохом освещении, при помощи нейронной сети.\
        Просто пришлите сюда фото!""",
    )


bot.polling(none_stop=True, interval=0)
