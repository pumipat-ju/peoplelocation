FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# We need to install torchreid. Usually it's installed via github or local clone.
# Assuming requirements.txt handles it, if not, we can add it here.
# Since the codebase uses ultralytics, torchreid, we ensure they are installed.

COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
