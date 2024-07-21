
import numpy as np
import torch
import cv2
import json
import pika
import os
import torch.nn.functional as F
import PIL
from torchvision import transforms
import base64
import pandas as pd


def to_input(pil_rgb_image):
    img = PIL.Image.fromarray(pil_rgb_image).convert("RGB")
    R, G, B = img.split()
    img = PIL.Image.merge("RGB", (B, G, R))
    data_transform = transforms.Compose([
    transforms.Resize([112,112]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = data_transform(img)
    return img 

# MagFace Functions
def load_pretrained_model():
    ckpt = torch.load('models/face.mag.unpg.pt', map_location=device)  # load checkpoint
    model_embed = ckpt['backbone'].to(device)
    return model_embed


def compute_embedding(channel, method, properties, body):

    body = json.loads(body)
    df = pd.DataFrame.from_dict(body)
    to_faces = []
    for _, row in df.iterrows():
        im = base64.b64decode(row['FaceImg'].encode("utf-8"))
        image_byte_array = bytearray(im)
        np_array = np.asarray(image_byte_array, dtype=np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        to_face = to_input(image)
        to_faces.append(to_face)

    imgs_tensor = torch.stack(to_faces).to("cuda")
    with torch.no_grad():
        features = model_magface(imgs_tensor)
        features = F.normalize(features)
    
    features = features.cpu().numpy().tolist()
    df.drop(['FaceImg'], axis=1, inplace=True)
    df.insert(1, 'Feature', features)
    df = df.to_json()
    channel.basic_publish(exchange='', routing_key=os.environ['EMB_QUEUE'], body=df)


def start_embedding():
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ['RABBITMQ_HOST']))
    channel = connection.channel()
    channel.queue_declare(queue=os.environ['FACE_QUEUE'])

    channel.basic_consume(queue=os.environ['FACE_QUEUE'], on_message_callback=compute_embedding, auto_ack=True)
    channel.start_consuming()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == "cuda":
    torch.cuda.empty_cache()
model_magface = load_pretrained_model()
start_embedding()


