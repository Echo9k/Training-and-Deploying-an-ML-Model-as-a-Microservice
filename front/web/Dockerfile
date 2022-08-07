FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential
WORKDIR /myapp
ADD ./requirements.txt /myapp
RUN pip3 install -r requirements.txt
ADD . /myapp
ENTRYPOINT ["python3"]
CMD ["app.py"]
