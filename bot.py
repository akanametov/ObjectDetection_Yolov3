import telebot
import torch
import urllib
import io
import os
from yolov3.yolov3_tiny import *
from dutils import PictureDetection

token='966235111:AAHBclJQQ_RuZVVA-YSlMXLAhoYOhceU88s'

bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start'])
def start_message(message):
    text="YoloDetectorBot \n[by Azamat Kanametov]\n\nSend me a photo..." 
    bot.send_message(message.chat.id, text)
    
def get_image_id(message):
    return message.photo[len(message.photo)-1].file_id

def save_image(message):
    image_id = get_image_id(message)
    bot.send_message(message.chat.id, 'Analyzing image, be patient...')

    file_path = bot.get_file(image_id).file_path

    image_url = "https://api.telegram.org/file/bot{0}/{1}".format(token, file_path)
    print(image_url)
    image_name = "images/{0}.jpg".format(image_id)
    urllib.request.urlretrieve(image_url, image_name)
    return image_name

def detect(message):
    path = save_image(message)
    model = Yolov3Tiny(num_classes=80)
    model.load_state_dict(torch.load('models/yolov3(tiny).h5'))
    img = PictureDetection(model, path, size=(256, 256), dataset='coco')
    Byte = io.BytesIO()
    img.save(Byte, format='PNG')
    imgByte = Byte.getvalue()
    os.remove(path)
    return imgByte

@bot.message_handler(content_types=['photo'])
def send_text(message):
    bot.send_photo(message.chat.id, detect(message))
    bot.send_message(message.chat.id, 'Objects detected!')

bot.polling()