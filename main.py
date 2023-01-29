import telebot
from PIL import Image
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
import numpy as np
from pix2pix import init_gen
import logging
from tools import get_image, resize_with_pad



logging.basicConfig(level=logging.DEBUG)

bot = telebot.TeleBot('5853232983:AAFYmhkJDd0L5IEIB0e_UbMO0KjPoKzmLuE')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = A.Compose(
    [A.Resize(256, 256), A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ), ToTensorV2()])
gen = init_gen()


@bot.message_handler(commands=["start"])
def start(m):
    bot.send_message(m.chat.id, 'Отправьте мне скетч персонажа из аниме, а я попробую его раскрасить:)')


@bot.message_handler(content_types=["text"])
def text(m):
    bot.send_message(m.chat.id, 'Я жду картиночку')


@bot.message_handler(content_types=['photo'])
def answer_to_photo(message):
    raw = message.photo[-1].file_id
    name = 'users_image/' + raw + ".jpg"
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)
    with open(name, 'wb') as new_file:
        new_file.write(downloaded_file)
    img_np = np.asarray(Image.open(name))
    img_np_resized = resize_with_pad(img_np, (286, 286))
    img = transform(image=img_np_resized)['image']
    painted_numpy = get_image(img, gen)
    painted_img = Image.fromarray(np.uint8(painted_numpy * 255))
    bot.send_photo(message.chat.id, painted_img)


bot.polling(none_stop=True, interval=0)
