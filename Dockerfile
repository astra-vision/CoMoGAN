FROM nvidia/cuda:11.0.3-base-ubuntu18.04

#Set zone
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update -q && \
    apt-get install -q -y \
        bzip2 \
        ca-certificates \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        mercurial \
        subversion \
        wget \
        nano \
        build-essential \
        cmake \
        unzip \
        pkg-config \
    && apt-get clean

ENV PATH /opt/conda/bin:$PATH

# Leave these args here to better use the Docker build cache
ARG CONDA_VERSION=py38_4.9.2
ARG CONDA_MD5=122c8c9beb51e124ab32a0fa6426c656

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    echo "${CONDA_MD5}  miniconda.sh" > miniconda.md5 && \
    if ! md5sum --status -c miniconda.md5; then exit 1; fi && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh miniconda.md5 && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

SHELL ["/bin/bash", "-c"]

RUN conda create -n comogan python=3.7.5 -y

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "comogan", "/bin/bash", "-c"]

RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y

RUN pip install visdom dominate tensorboard==2.3.0 tensorflow==2.3.0 coverage && \
    pip install munch pytorch-lightning==1.1.8 human-id && \
    pip install jupyter waymo-open-dataset-tf-2-3-0 imageio imageio-ffmpeg
    
EXPOSE 8888
    
RUN git clone https://github.com/cv-rits/CoMoGAN.git

WORKDIR CoMoGAN

RUN conda init && \
    echo "conda activate comogan" >> /root/.bashrc && \
    echo "jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --no-browser ." > /root/.bash_history && \
    echo "python3 train.py" >> /root/.bash_history

CMD ["/bin/bash"]
