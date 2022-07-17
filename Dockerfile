FROM python:3.7-slim

MAINTAINER canwang<wangcan1213@gmail.com>

ADD . .

RUN pip install -r requirements.txt -i http://mirrors.tencentyun.com/pypi/simple --trusted-host mirrors.tencentyun.com

# RUN pip install -r requirements.txt

WORKDIR /frontend

RUN apt update

RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

EXPOSE 5000

CMD ["python", "app.py"]

