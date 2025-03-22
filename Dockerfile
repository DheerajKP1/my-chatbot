FROM python:3.10-slim

WORKDIR /my-app

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Then copy the rest of your application
COPY . .

EXPOSE 8000

CMD chainlit run mem-app.py --host 0.0.0.0 --port 8000 -w