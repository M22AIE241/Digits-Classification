# #pull latest version from dockerhub
# # FROM ubuntu:23.10
# FROM python:3.9.17
# #run docker build -f Dockerfile .
# #run docker images to see the existing and downloaded images
# #if wnat to give name to the images pulled
# #docker build -t ubuntu_latest:v1 -f Dockerfile
# #when only running this images created , a container will be created that will run the contents and executebles of this image
# #once the images will stop running , the container will collapse
# #hence container creating from images to run the commands of images
# COPY . /digits/
# # RUN apt-get update
# # RUN apt-get install -y python3 python3-pip
# # RUN apt-get install -y pip3 
# RUN pip3 install -r /digits/requirements.txt
# CMD ["pytest"]
FROM python:3.9.17
# copy code folder
COPY . /digits/
# RUN apt-get update
# RUN apt-get install -y python3 python3-pip
RUN pip3 install -r /digits/requirements.txt
#set work dir
WORKDIR /digits
#create volume to mount it on host
VOLUME /digits/models
#run pytest 
#CMD ["pytest"]
ENV FLASK_APP=api/app2.py
CMD ["flask","run","--host=0.0.0.0"]
#run python script to train model
# ENTRYPOINT ["python","exp.py"]
