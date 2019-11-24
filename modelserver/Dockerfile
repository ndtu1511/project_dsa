FROM jjanzic/docker-python3-opencv

WORKDIR /app

COPY requirements.txt requirements.txt 

RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "main.py"]