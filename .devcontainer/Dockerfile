# Use Docker stable as the base image
FROM docker:stable

# Install dependencies
RUN apk add --no-cache \
    bash \
    curl \
    git \
    ca-certificates

# Install Tilt
RUN curl -fsSL https://github.com/tilt-dev/tilt/releases/download/v0.33.17/tilt.0.33.17.linux.x86_64.tar.gz | tar -xzv tilt && \
    mv tilt /usr/local/bin/tilt

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x ./kubectl && \
    mv ./kubectl /usr/local/bin/kubectl

# Install kind
RUN curl -Lo ./kind "https://kind.sigs.k8s.io/dl/v0.11.1/kind-linux-amd64" && \
    chmod +x ./kind && \
    mv ./kind /usr/local/bin/kind

# Install ctlptl
RUN curl -Lo ctlptl.tar.gz "https://github.com/tilt-dev/ctlptl/releases/download/v0.5.7/ctlptl_0.5.7_linux_amd64.tar.gz" && \
    tar -xzvf ctlptl.tar.gz ctlptl && \
    mv ctlptl /usr/local/bin/ctlptl && \
    rm ctlptl.tar.gz

# Setup micromamba
RUN curl -L -O "https://micromamba.snakepit.net/api/micromamba/linux-64/latest" && \
    mv latest micromamba && \
    chmod +x micromamba && \
    ./micromamba shell init -s bash -p ~/micromamba && \
    rm micromamba && \
    echo 'micromamba create -n dev-env python=3.10 uv -c conda-forge' >> ~/.bashrc

# Cleanup
RUN rm -rf /var/cache/apk/* && \
    rm -rf /tmp/*

# Set the default command
CMD ["bash"]