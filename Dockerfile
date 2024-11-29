FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make the script executable
RUN chmod +x app.py

# Use ENTRYPOINT with CMD for better command handling
ENTRYPOINT ["python"]
CMD ["app.py"] 