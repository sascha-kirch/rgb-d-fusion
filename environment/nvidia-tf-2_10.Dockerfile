FROM nvcr.io/nvidia/tensorflow:22.02-tf2-py3

USER 1000:1000

# Copy the requirements.txt file to our Docker image
ADD requirements.txt .

# Install packages
RUN apt update && \
apt install ffmpeg libsm6 libxext6 graphviz -y && \
pip install -r requirements.txt