FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get upgrade -y
RUN pip --timeout=1000 install --no-cache-dir --upgrade flask docker
COPY . /app
CMD ["python", "app.py"]