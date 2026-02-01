# Multi-stage build
FROM python:3.10-alpine AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-alpine
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY . .

# Create non-root user and set permissions
RUN adduser -D -u 1000 ultimate && \
    chown -R ultimate:ultimate /app

USER ultimate

EXPOSE 8080

CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8080", "main:app"]