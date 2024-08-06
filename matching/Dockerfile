FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get upgrade -y
RUN pip --timeout=1000 install --no-cache-dir --upgrade numpy pika
RUN pip install pandas
RUN pip install psycopg2-binary
COPY . /app
CMD ["python", "matching.py"]