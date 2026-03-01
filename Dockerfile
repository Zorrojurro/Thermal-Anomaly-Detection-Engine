FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 git git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than GPU version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HuggingFace Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "web_app.py"]
