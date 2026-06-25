FROM intel/oneapi-basekit:2026.0.0-0-devel-ubuntu22.04

ARG PYTHON_VERSION=3.12

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

RUN pip install --pre torch==2.14.0.dev20260624+xpu --index-url https://download.pytorch.org/whl/nightly/xpu

WORKDIR /workspace

CMD ["/bin/bash"]
