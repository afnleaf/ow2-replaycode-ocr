FROM python:3.11-slim

WORKDIR /app

# Get TROCR model cache
#COPY model.safetensors /app/model-data/
#RUN --mount=type=cache,target=/app/model-data

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopencv-dev \
    tesseract-ocr

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

CMD [ "python", "./src/main.py" ]
#CMD [ "python", "trocr-test.py" ]
#CMD [ "python", "./src/testtrocr.py" ]
#CMD [ "python", "./src/ocr.py" ]
#CMD [ "python", "./src/test.py" ]

