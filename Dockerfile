FROM python:3.12-slim

WORKDIR /app

# Install CPU-only PyTorch first (much smaller than the default CUDA build)
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

CMD ["gunicorn", "--preload", "--workers", "1", "--threads", "4", \
     "--timeout", "120", "--bind", "0.0.0.0:8000", "app:app"]
