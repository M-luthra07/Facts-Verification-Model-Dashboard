FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Railway
EXPOSE 5000

# Start the server
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]

