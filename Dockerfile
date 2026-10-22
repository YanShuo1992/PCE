FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /dgm

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]
