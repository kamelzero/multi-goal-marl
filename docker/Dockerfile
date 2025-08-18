FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg make && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command runs a tiny demo (override in make)
CMD ["bash", "-lc", "make demo"]


