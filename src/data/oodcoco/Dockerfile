FROM ubuntu:latest

ARG python_version=3.8.3

# For apt to work through proxy.
RUN echo "Acquire::http::Proxy \"$HTTP_PROXY\";" >> /etc/apt/apt.conf
RUN echo "Acquire::https::Proxy \"$HTTPS_PROXY\";" >> /etc/apt/apt.conf

RUN ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime

# For pyenv.
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -qqy make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl \
    git && rm -rf /var/lib/apt/lists/*

RUN dpkg-reconfigure -f noninteractive tzdata

RUN echo "export HTTP_PROXY=$HTTP_PROXY" >> ~/.bashrc
RUN echo "export HTTPS_PROXY=$HTTPS_PROXY" >> ~/.bashrc
RUN curl https://pyenv.run | bash

ENV PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/shims/:$PYENV_ROOT/bin:$PATH
ENV PYENV_VERSION=$python_version

ARG python=/usr/local/python-$python_version

RUN ~/.pyenv/plugins/python-build/bin/python-build $python_version $python
ENV PATH=$python/bin:$PATH

# Install dependencies.
RUN pip --no-cache-dir install --upgrade pip && pip --no-cache-dir install pycocotools numpy pillow scipy