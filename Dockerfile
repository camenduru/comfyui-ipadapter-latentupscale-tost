FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    scipy==1.13.1 torchsde==0.2.6 einops==0.8.0 transformers==4.41.2 diffusers==0.28.0 accelerate==0.30.1 xformers==0.0.25

RUN git clone -b totoro https://github.com/camenduru/ComfyUI /content/TotoroUI
RUN git clone -b totoro https://github.com/camenduru/ComfyUI_IPAdapter_plus /content/TotoroUI/IPAdapter

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/128713 -d /content/TotoroUI/models -o dreamshaper_8.safetensors
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors -d /content/TotoroUI/models/clip_vision -o CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors -d /content/TotoroUI/models/ipadapter -o ip-adapter-plus_sd15.safetensors
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/Ttl/ComfyUi_NNLatentUpscale/raw/master/sd15_resizer.pt -d /content/TotoroUI/models -o sd15_resizer.pt

COPY ./worker_runpod.py /content/TotoroUI/worker_runpod.py
WORKDIR /content/TotoroUI
CMD python worker_runpod.py
