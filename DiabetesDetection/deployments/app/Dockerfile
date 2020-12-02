FROM python:3.6
ADD app/ /app
WORKDIR /app
RUN apt-get update
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
CMD python Main.py
