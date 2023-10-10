FROM tensorflow/tensorflow:2.10.0-gpu-jupyter

# Copy the requirements.txt file to our Docker image
ADD requirements.txt .

# Prequesites for open CV
RUN apt update
RUN apt install ffmpeg libsm6 libxext6  -y

RUN apt install graphviz -y

RUN apt install dos2unix

# Install the requirements.txt
RUN pip install -r requirements.txt