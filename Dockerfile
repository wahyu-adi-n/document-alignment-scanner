FROM jjanzic/docker-python3-opencv:opencv-3.2.0

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY . /app

COPY templates templates

EXPOSE 5000

ENTRYPOINT [ "python3"]

CMD ["app.py"]