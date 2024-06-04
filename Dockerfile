# Use an Ubuntu base image
FROM python:3.10

# working directory
WORKDIR /app

# Install OpenGL/graphics libs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip 

ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

# Install dependencies required for Node.js and NVM
RUN python3 -m pip install --user --upgrade pip

RUN pip3 install torch torchvision torchaudio

RUN pip install -U ninja transformers accelerate flask flask_cors Pillow peft sentencepiece timm packaging einops

RUN ldconfig -v

# Copying the rest of the application code
COPY . .

# Exposing the port app runs
EXPOSE 5555

# Start the application with npm run dev
CMD ["python", "server.py"]


