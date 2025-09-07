# Step 1: Start with an official Python base image
FROM python:3.11-slim

# Step 2: Set the main working directory inside the server
WORKDIR /app

# Step 3: Update the server's package list and install FFmpeg
# This is the crucial step for your audio processing to work
RUN apt-get update && apt-get install -y ffmpeg

# Step 4: Copy your requirements file into the server
COPY requirements.txt .

# Step 5: Install all of your Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy your entire project folder into the server
COPY . .

# Step 7: Tell the server that your app will run on port 8000
EXPOSE 8000

# Step 8: Define the command to start your app using the gunicorn server
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:8000", "app:app"]