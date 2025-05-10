# Dockerfile
FROM python:3.11-slim

# system deps for OpenCV, etc.
RUN apt-get update && apt-get install -y \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy & install requirements + socketio server
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask-socketio eventlet

# copy source
COPY . .

# expose the ngrok port
EXPOSE 4150

# run via the Flask-SocketIO eventlet server
CMD ["python", "server.py"]

