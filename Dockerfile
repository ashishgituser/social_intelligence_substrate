FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cache-friendly layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
