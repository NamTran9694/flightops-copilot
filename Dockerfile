# FlightOps Copilot - one image runs the Streamlit app (default) or the API
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# build the synthetic database inside the image, so the container starts ready
RUN python scripts/generate_data.py

EXPOSE 8501 8000
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
