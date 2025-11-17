# Stage 1
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# Stage 2
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/ ./src
COPY smsspamcollection/ ./smsspamcollection

RUN mkdir -p output

# F6: configurable port
ENV SERVER_PORT=${SERVER_PORT:-8081}
EXPOSE ${SERVER_PORT}

CMD ["python", "src/serve_model.py"]
