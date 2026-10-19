# PDF Research Assistant — portable container image.
# Build:  docker build -t pdf-assistant .
# Run:    docker run -p 8501:8501 --env-file .env -v pdf_data:/app/data pdf-assistant
FROM python:3.11-slim

# PyMuPDF ships manylinux wheels, so no system build deps are needed.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Persist uploaded PDFs + index across restarts (mount a volume here).
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py"]
