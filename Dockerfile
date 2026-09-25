FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# reproduce all results + figures + tables, then start the demo gateway
CMD ["sh", "-c", "python run_experiments.py && python make_figures.py && python make_tables.py && uvicorn demo.app:app --host 0.0.0.0 --port 8000"]
