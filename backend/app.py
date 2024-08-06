from flask import Flask, request, jsonify, render_template, redirect, url_for
import docker
import logging

app = Flask(__name__)
client = docker.from_env()

logging.basicConfig(level=logging.WARNING)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera-view/<int:camera_number>')
def camera_view(camera_number):
    return render_template('camera_view.html', camera_number=camera_number)

@app.route('/start-camera', methods=['POST'])
def start_camera():
    data = request.json
    camera_number = data['cameraNumber']
    rtsp_url = data['rtspUrl']

    container_name = f'vid2frame_{camera_number}'

    try:
        # Check if container already exists
        try:
            existing_container = client.containers.get(container_name)
            existing_container.stop()
            existing_container.remove()
            logging.debug(f"Existing container {container_name} stopped and removed.")
        except docker.errors.NotFound:
            logging.debug(f"No existing container named {container_name} found.")

        logging.debug(f"Starting container {container_name} with RTSP URL: {rtsp_url}")

        container = client.containers.run(
            "new_robin-vid2frame",
            detach=True,
            environment={
                'VID_SOURCE': rtsp_url,
                'VID_FPS': 5,
                'CAM_NUMBER': camera_number,
                'RABBITMQ_HOST': 'rabbitmq',
                'FRAME_QUEUE': f'frame_queue_{camera_number}'
            },
            name=container_name,
            volumes={
                '/home/next/Documents/Projects/shared': {'bind': '/shared', 'mode': 'rw'}
            },
            restart_policy={"Name": "on-failure"},
            network='new_robin_default',
            ports={
                '8765/tcp': 8000
            }
        )

        logging.debug(f"Container {container_name} started successfully.")
        return jsonify({"message": f"Camera {camera_number} started successfully!"}), 200
    except Exception as e:
        logging.error(f"Failed to start camera {camera_number}: {str(e)}")
        return jsonify({"message": f"Failed to start camera {camera_number}: {str(e)}"}), 500

@app.route('/stop-camera', methods=['POST'])
def stop_camera():
    data = request.json
    camera_number = data['cameraNumber']

    container_name = f'vid2frame_{camera_number}'

    try:
        container = client.containers.get(container_name)
        container.stop()
        container.remove()
        logging.debug(f"Container {container_name} stopped and removed successfully.")
        return jsonify({"message": f"Camera {camera_number} stopped successfully!"}), 200
    except Exception as e:
        logging.error(f"Failed to stop camera {camera_number}: {str(e)}")
        return jsonify({"message": f"Failed to stop camera {camera_number}: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
