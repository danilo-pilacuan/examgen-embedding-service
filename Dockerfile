FROM python:3.11-slim

WORKDIR /app

# Install ONLY CPU torch first (avoids downloading hundreds of MB of CUDA).
# Then sentence-transformers with its normal dependencies: since torch is
# already installed, pip will not reinstall it with CUDA.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn \
        sentence-transformers

COPY src/embedding_service/app.py /app/embedding_service_app.py

EXPOSE 8081

CMD ["uvicorn", "embedding_service_app:app", "--host", "0.0.0.0", "--port", "8081"]
