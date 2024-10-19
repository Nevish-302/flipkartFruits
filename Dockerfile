FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y curl

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    python3.9 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./
COPY ./server.py ./
COPY ./tester.py ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["flask", "--app", "server.py", "run", "-h", "0.0.0.0", "-p", "8000"]