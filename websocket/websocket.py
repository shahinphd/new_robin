import asyncio
import websockets
import json
import os
import pika

# Global variable to indicate if the client has finished
client_finished = False

async def rabbitmq_consumer(websocket, queue_name):
    global client_finished

    # Connect to RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.environ['RABBITMQ_HOST']))
    channel = connection.channel()

    # Declare the queue
    channel.queue_declare(queue=queue_name)

    # Define a callback function to handle messages from RabbitMQ
    def callback(ch, method, properties, body):
        if not client_finished:
            # Send the message to the WebSocket client
            asyncio.run_coroutine_threadsafe(websocket.send(body), asyncio.get_event_loop())
        else:
            ch.stop_consuming()

    # Start consuming messages from RabbitMQ
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    channel.start_consuming()

async def handle_client(websocket, path):
    global client_finished
    client_finished = False

    # Wait for the initial message from the client
    cam_id = await websocket.recv()
    print(f"Received cam_id: {cam_id}")

    # Start the RabbitMQ consumer in a separate thread
    rabbitmq_task = asyncio.create_task(rabbitmq_consumer(websocket, 'MATCHED_QUEUE'))

    if websockets.ConnectionClosed:
        print("Client disconnected.")
        rabbitmq_task.cancel()

async def main():
    # Start the WebSocket server
    async with websockets.serve(handle_client, "localhost", 8876):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
