FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /app
RUN apt-get update && apt-get upgrade -y
RUN pip --timeout=1000 install --no-cache-dir --upgrade opencv-python
RUN pip --timeout=1000 install --no-cache-dir --upgrade pandas numpy pika
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY . /app
CMD ["python", "detector.py"]