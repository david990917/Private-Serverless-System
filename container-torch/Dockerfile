FROM python:3.6-slim

COPY pip.conf /etc/pip.conf


RUN apt-get -y update && \
    apt-get -y upgrade


# fulfill the structure requirement of proxy
RUN mkdir /proxy && \
    mkdir /proxy/exec

# copy the proxy server
COPY proxy.py /proxy/

# the work dir of proxy is under exec/
WORKDIR /proxy/exec

# proxy server runs under port 5000
EXPOSE 5000

RUN pip3 install --no-cache-dir gevent flask torch torchvision pillow urllib3

CMD ["python3", "/proxy/proxy.py"]