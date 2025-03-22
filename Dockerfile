FROM python:3.10-slim

WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install additional packages needed for HF compatibility
RUN pip install langchain-community langchain-huggingface

# Then copy the rest of your application
COPY . .

# Hugging Face Spaces typically uses port 7860
EXPOSE 7860

# Command that Hugging Face expects
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]

# FROM python:3.10-slim
# RUN useradd -m -u 1000 user
# USER user
# ENV HOME=/home/user \
#     PATH=/home/user/.local/bin:$PATH
# WORKDIR $HOME/app
# COPY --chown=user . $HOME/app
# COPY ./requirements.txt ~/app/requirements.txt
# RUN pip install -r requirements.txt
# COPY . .
# CMD ["chainlit", "run", "app.py", "--port", "7860"]