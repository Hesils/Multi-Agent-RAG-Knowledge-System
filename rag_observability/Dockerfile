FROM python:3.14

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn prometheus-client

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]