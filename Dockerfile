# 1. Use a slim Python 3.11 base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y gcc libffi-dev build-essential

# 4. Copy project files to the container
COPY . .

# 5. Install Python packages from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 6. Ensure Railway passes environment variables (no hardcoding needed)
ENV PYTHONUNBUFFERED=1

# 7. Run Dramatiq worker when the container starts
CMD ["python", "-m", "dramatiq", "jobs", "--watch=.", "--threads=4"]
