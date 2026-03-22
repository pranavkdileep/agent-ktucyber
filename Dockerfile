
FROM python:3.12.3-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the project
COPY . /app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "chat_api:app", "--host", "0.0.0.0", "--port", "8000"]
