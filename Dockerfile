# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set environment variable (optional: actual value will be taken from Railway)
ENV REDIS_URL=${REDIS_URL}

# Start Dramatiq worker when container starts
CMD ["python", "-m", "dramatiq", "jobs"]
