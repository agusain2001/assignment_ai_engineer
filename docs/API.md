# API Documentation

## Endpoints

### POST /analyze
Analyze a single message for safety concerns.

**Request:**
```json
{
    "text": "Message to analyze",
    "user_age": 16  // Optional
}
```

**Response:**
```json
{
    "success": true,
    "analysis": {
        "risk_level": "low",
        "risk_score": 0.25,
        "interventions": ["warning"]
    }
}
```

### POST /batch_analyze
Analyze multiple messages.

### GET /stats
Get analysis statistics.

### GET /health
Health check endpoint.
