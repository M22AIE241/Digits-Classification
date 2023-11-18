FROM python:3.8
WORKDIR /app
COPY app.py /app/
#COPY utils.py /app/
COPY requirements.txt /app/
COPY saved_model /app/saved_model/
RUN apt-get install libsm6 libxext6  -y
RUN pip install -r requirements.txt
VOLUME /app/saved_model
EXPOSE 80
CMD ["python", "app.py"]