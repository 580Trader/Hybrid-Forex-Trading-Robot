FROM python:3.11-slim

WORKDIR /app

# Install python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the rest of the code and data
COPY . .

# Run the deterministic runner
CMD ["python", "-u", "zerocore_runner.py"]
