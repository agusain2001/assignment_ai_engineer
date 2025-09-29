# Deployment Guide

## Local Deployment

1. Install dependencies: `pip install -r requirements.txt`
2. Train models: `python scripts/train_models.py`
3. Run server: `python web_app/app.py`

## Docker Deployment

```bash
docker build -t ai-safety .
docker run -p 5000:5000 ai-safety
```

## Production Deployment

- Use Gunicorn/uWSGI for WSGI server
- Deploy behind nginx/Apache
- Use Redis for caching
- PostgreSQL for production database
