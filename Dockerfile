# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set default environment variable (optional - Railway injects it at runtime)
# You can safely remove this line if Railway manages the env vars.
# ENV REDIS_URL=rediss://default:<your-pass>@romantic-skylark-25427.upstash.io:6379

# Run Dramatiq when the container starts
CMD ["python", "-m", "dramatiq", "jobs", "--processes", "4", "--threads", "4"]
