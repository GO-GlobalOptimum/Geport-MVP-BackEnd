# Use an official Python runtime as a parent image
FROM python:3.9

# Copy the current directory contents into the container at /app
COPY . /src
WORKDIR /src

RUN apt-get update && apt-get install -y sqlite3
RUN pip install -r requirements.txt -v

EXPOSE 8000


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
