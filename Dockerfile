FROM python:3.12-slim

WORKDIR /app

# Install CPU-only PyTorch first (much smaller than the default CUDA build)
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# Use sh -c so ${PORT} expands at runtime — Render injects PORT=10000
# and other platforms (Railway, Fly) do similar. Falls back to 8000 locally.
CMD ["sh", "-c", "gunicorn --preload --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-8000} app:app"]