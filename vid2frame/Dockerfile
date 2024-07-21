FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip --timeout=1000 install --no-cache-dir --upgrade opencv-python
RUN pip --timeout=1000 install --no-cache-dir --upgrade ffmpegio pika
WORKDIR /app
COPY . /app
CMD ["python", "vid2frame.py"]