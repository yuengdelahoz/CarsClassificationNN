FROM tensorflow/tensorflow:latest-devel-gpu-py3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

CMD [ "python3", "-u", "./train.py"]
