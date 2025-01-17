# Use the official Python 3.12-slim image as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by some Python libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libmysqlclient-dev

# Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Install the Python dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy the model and logs to the container (optional)
COPY ./models /app/models
COPY ./logs /app/logs

# Expose port 5000 for the MLflow REST API
EXPOSE 5000

# Command to start the MLflow model server
CMD ["mlflow", "models", "serve", "--model-uri", "file:/app/models", "--host", "0.0.0.0", "--port", "5001"]
