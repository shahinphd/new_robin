import ffmpegio
import pika
import os
import cv2
import base64
import json
import asyncio
import websockets

async def stream_frames(websocket):
    video_file = os.environ['VID_SOURCE']
    fps = int(os.environ['VID_FPS'])
    cam_number = os.environ['CAM_NUMBER']
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ['RABBITMQ_HOST']))
    channel = connection.channel()
    channel.queue_declare(queue=os.environ['FRAME_QUEUE'])

    with ffmpegio.open(video_file, 'rv', r=fps) as f:
        for frame in f:
            frame = cv2.cvtColor(frame[0],cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', frame)
            img_encode = base64.b64encode(buffer).decode('utf-8')
            body = {'cam_id': cam_number, 'frame': img_encode}
            channel.basic_publish(exchange='', routing_key=os.environ['FRAME_QUEUE'], body=json.dumps(body))
            await websocket.send(json.dumps(body))
    connection.close()

async def handler(websocket, path):
    await stream_frames(websocket)

start_server = websockets.serve(handler, '0.0.0.0', 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
