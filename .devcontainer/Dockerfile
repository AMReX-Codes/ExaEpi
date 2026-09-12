FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Install build deps and Python packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    cmake git git-lfs g++ make ninja-build \
    libopenmpi-dev openmpi-bin libhdf5-openmpi-dev \
    ccache python3 python3-pip clang-format sudo \
    curl ca-certificates poppler-utils \
    && pip3 install --no-cache-dir --break-system-packages \
       numpy matplotlib pandas pandas-stubs h5py scipy \
       geopandas yt polars \
       psutil colorama censusgeocode \
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Rename the existing ubuntu user/group (UID/GID 1000) to match USERNAME
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupmod --new-name ${USERNAME} ubuntu \
    && usermod --login ${USERNAME} --move-home --home /home/${USERNAME} ubuntu \
    # Allow passwordless sudo
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Ensure ccache and Claude Code config directories are owned by the user
# (these get overlaid by named volumes in devcontainer.json, which seed their
# initial ownership from whatever exists here at first mount)
RUN mkdir -p /home/${USERNAME}/.ccache /home/${USERNAME}/.claude \
    && chown -R ${USER_UID}:${USER_GID} /home/${USERNAME}/.ccache /home/${USERNAME}/.claude

USER ${USERNAME}
