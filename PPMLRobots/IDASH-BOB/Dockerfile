FROM ubuntu
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
RUN apt-get install -y net-tools
RUN apt-get install vim -y
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y iputils-ping
RUN apt-get install -y telnet
COPY main.py /
COPY numpysocket.py /
COPY processData.py /
COPY trainIDASH.py /
COPY inferIDASH.py /
COPY party_1.csv /
COPY requirements.txt /
RUN pip install -r requirements.txt


