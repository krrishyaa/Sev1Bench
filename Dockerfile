FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860

WORKDIR /app

COPY requirements.txt inference.py models.py /app/
COPY server /app/server

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["sh", "-c", "uvicorn server.app:app --host ${HOST} --port ${PORT}"]