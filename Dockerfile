FROM python:3.12-slim

# Hugging Face Spaces convention: run as a non-root user with UID 1000.
# Without this, the HF Hub cache directory and any writes the runtime
# performs end up owned by root and the Space's volume layer rejects them.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install CPU-only PyTorch first (much smaller than the default CUDA build).
# --user keeps everything under /home/user/.local so it's writable by `user`.
RUN pip install --no-cache-dir --user torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user . .

# HF Spaces routes external traffic to whatever port `app_port` in the
# README YAML specifies; default is 7860. We honor $PORT if set so the
# same image still works locally / on Render / on Railway.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "gunicorn --preload --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-7860} app:app"]