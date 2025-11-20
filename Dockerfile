# Stage 1: Build and train model
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY ./requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# Copy source code and data
COPY ./src/ ./src

# Install packages to system location and train the model
RUN pip install -r requirements.txt

# Train the model to generate required files
ENV PYTHONPATH=/app/src

# Stage 2: Runtime
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /app/src ./src

# F6: configurable port
ENV SERVER_PORT=${SERVER_PORT:-8081}
EXPOSE ${SERVER_PORT}

CMD ["python", "src/serve_model.py"]
