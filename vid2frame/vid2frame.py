import ffmpegio
import pika
import os
import shutil
import cv2
import base64
import json

def vid2frame(video_file, fps, duration=None):
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ['RABBITMQ_HOST']))
    channel = connection.channel()
    channel.queue_declare(queue=os.environ['FRAME_QUEUE'])
    channel.queue_declare(queue=os.environ['STREAM_QUEUE'])
    
    print(os.environ['FRAME_QUEUE'], flush=True)
    print(os.environ['STREAM_QUEUE'], flush=True)

    if duration is not None:
        with ffmpegio.open(video_file, 'rv', t=duration, r=fps) as f:
            for frame in f:
                _, buffer = cv2.imencode('.jpg', frame[0])
                channel.basic_publish(exchange='', routing_key=os.environ['FRAME_QUEUE'], body=buffer.tobytes())
                channel.basic_publish(exchange='', routing_key=os.environ['STREAM_QUEUE'], body=buffer.tobytes())
        connection.close()

    else:
        with ffmpegio.open(video_file, 'rv', r=fps) as f:
            for frame in f:
                _, buffer = cv2.imencode('.jpg', frame[0])
                img_encode = base64.b64encode(buffer).decode('utf-8')
                body = {'cam_id': os.environ['CAM_NUMBER'], 'frame': img_encode}
                channel.basic_publish(exchange='', routing_key=os.environ['FRAME_QUEUE'], body=json.dumps(body))
                channel.basic_publish(exchange='', routing_key=os.environ['STREAM_QUEUE'], body=json.dumps(body))
        connection.close()

if os.path.exists("/shared/frames"):
    shutil.rmtree("/shared/frames")
os.mkdir("/shared/frames")

if os.path.exists("/shared/faces"):
    shutil.rmtree("/shared/faces")
os.mkdir("/shared/faces")

vid2frame(os.environ['VID_SOURCE'], os.environ['VID_FPS'])
