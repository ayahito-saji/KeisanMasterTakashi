FROM python:3.6

WORKDIR /takashi

RUN pip install torch==1.7.1

CMD bash