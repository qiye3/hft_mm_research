FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements-research.txt backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements-research.txt
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scripts/make_demo_dataset.py ./scripts/
ENV PYTHONPATH=/app/backend
ENV FACTORBOARD_STATE_DIR=/app/workspace
ENV FACTORBOARD_DATA_DIR=/datasets/market_groups_v2
ENV FACTORBOARD_MAX_JOBS=2
EXPOSE 8000
CMD ["uvicorn","factorboard_api.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
