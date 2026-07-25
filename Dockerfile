# Use a lightweight official Python image
FROM python:3.11-slim

# Set the workspace directory inside the container
WORKDIR /app

# Copy dependency definitions and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API source code into the container
COPY app.py .

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the production web server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]