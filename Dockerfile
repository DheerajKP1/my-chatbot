FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

EXPOSE 7860
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