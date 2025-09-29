FROM python:3.9-slim

WORKDIR /app

COPY requirements-file.txt .
RUN pip install --no-cache-dir -r requirements-file.txt

COPY . .

RUN python scripts/train_models.py

EXPOSE 5000

CMD ["python", "web_app/app.py"]
