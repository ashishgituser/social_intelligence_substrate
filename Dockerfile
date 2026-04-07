FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cache-friendly layer)
COPY requirements.txt .
RUN pip install --no-cache-dir openenv-core>=0.2.0 && \
    pip install --no-cache-dir networkx==3.2.1 python-dotenv==1.0.0 httpx==0.25.2 pytest==7.4.4

# Copy application code
COPY . .

# Hugging Face Spaces expects port 7860
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
