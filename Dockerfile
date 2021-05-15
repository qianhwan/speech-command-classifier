FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y libsndfile1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY speech_command_classifier/ speech_command_classifier/
COPY setup.py .
RUN pip install -e .

WORKDIR /home/workspace
