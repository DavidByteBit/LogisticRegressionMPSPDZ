apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

apt-get install -y net-tools
apt-get install vim -y
apt-get install -y libsm6 libxext6 libxrender-dev
apt-get install -y iputils-ping
apt-get install -y telnet
pip install -r /requirements.txt
chmod +x /main.py