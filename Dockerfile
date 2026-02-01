# Multi-stage build
FROM python:3.10-alpine AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.10-alpine
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
RUN adduser -D -u 1000 xtream && chown -R xtream:xtream /app
USER xtream
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8080
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8080", "main:app"]