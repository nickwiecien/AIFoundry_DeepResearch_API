# Deep Research & Sora API

A FastAPI-based REST API that exposes Azure AI Deep Research and OpenAI Sora video generation capabilities.

## Features

- **Deep Research**: Conduct comprehensive research using Azure AI with Bing search integration
- **Video Generation**: Generate videos using OpenAI's Sora model via Azure OpenAI
- **Asynchronous Processing**: All operations run as background jobs with status tracking
- **File Downloads**: Download research reports (MD/PDF) and generated videos
- **Job Management**: Track and monitor all running jobs

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   Copy `.env.example` to `.env` and fill in your Azure credentials:
   ```bash
   cp .env.example .env
   ```

3. **Set up Azure authentication**:
   - Install Azure CLI: `az login`
   - Or configure service principal credentials in `.env`

## Running the API

Start the development server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- **Interactive docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check
- `GET /health` - Check API health status

### Deep Research
- `POST /research/start` - Start a research job
- `GET /research/{job_id}/status` - Get research job status
- `GET /research/{job_id}/download?format=md|pdf` - Download research report

### Video Generation
- `POST /video/generate` - Start video generation job
- `GET /video/{job_id}/status` - Get video generation job status
- `GET /video/{job_id}/download` - Download generated video

### Job Management
- `GET /jobs` - List all jobs and their status

## Example Usage

### Start Deep Research
```bash
curl -X POST "http://localhost:8000/research/start" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Latest developments in quantum computing",
    "max_turns": 5
  }'
```

### Start Video Generation
```bash
curl -X POST "http://localhost:8000/video/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing piano in a jazz bar",
    "width": 480,
    "height": 480,
    "n_seconds": 5
  }'
```

### Check Job Status
```bash
curl "http://localhost:8000/research/{job_id}/status"
```

### Download Results
```bash
# Download research report as PDF
curl "http://localhost:8000/research/{job_id}/download?format=pdf" -o report.pdf

# Download generated video
curl "http://localhost:8000/video/{job_id}/download" -o video.mp4
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `PROJECT_ENDPOINT` | Azure AI Project endpoint | Yes |
| `MODEL_DEPLOYMENT_NAME` | GPT-4 model deployment name | Yes |
| `DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME` | Deep research model deployment | Yes |
| `BING_RESOURCE_NAME` | Bing connection resource name | Yes |
| `AOAI_ENDPOINT` | Azure OpenAI endpoint | Yes |
| `AOAI_KEY` | Azure OpenAI API key | Yes |
| `AZURE_CLIENT_ID` | Service principal client ID | Optional* |
| `AZURE_CLIENT_SECRET` | Service principal secret | Optional* |
| `AZURE_TENANT_ID` | Azure tenant ID | Optional* |

*Required only if not using Azure CLI or Managed Identity authentication

## File Structure

```
api/
├── main.py              # FastAPI application
├── research_utils.py    # Research utility functions
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── reports/            # Generated research reports (created at runtime)
├── videos/             # Generated videos (created at runtime)
└── README.md          # This file
```

## Response Formats

### Job Response
```json
{
  "job_id": "uuid-string",
  "status": "started|running|completed|failed",
  "created_at": "2025-01-01T12:00:00"
}
```

### Job Status Response
```json
{
  "job_id": "uuid-string",
  "status": "started|running|completed|failed",
  "progress": "Current progress description",
  "result": {
    "report_available": true,
    "formats": ["md", "pdf"]
  },
  "error": null
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Not Found (job doesn't exist)
- `500` - Internal Server Error

## Production Deployment

For production deployment:

1. Use a production WSGI server like Gunicorn
2. Set up proper logging
3. Use a database for job storage instead of in-memory storage
4. Configure proper authentication and authorization
5. Set up file storage (Azure Blob Storage, S3, etc.)
6. Use environment-specific configuration files

## Troubleshooting

1. **Authentication Issues**: Ensure Azure credentials are properly configured
2. **Model Not Found**: Verify model deployment names in Azure AI Studio
3. **Bing Connection**: Ensure Bing Search resource is connected to your AI project
4. **File Permissions**: Ensure the API has write permissions for reports/ and videos/ directories
