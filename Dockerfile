# Use a slim Python image to keep the container small
FROM python:3.11-slim

# Create a non-root user (Hugging Face requirement: UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 7860

# Set the working directory
WORKDIR /app

# Install system dependencies for OpenCV (run as root, then switch back to user)
USER root
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy the requirements file and install dependencies
# We specifically use the CPU version of torch to keep the image size low
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project with correct ownership
COPY --chown=user . /app

# Expose the port (Hugging Face uses 7860 by default)
EXPOSE 7860

# Command to run the application using uvicorn
# Note: Ensure the path matches your project structure (backend.main:app)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
