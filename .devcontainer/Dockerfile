FROM mcr.microsoft.com/devcontainers/base:bookworm
ARG PIXI_VERSION=latest

# derived from https://pixi.sh/latest/ide_integration/devcontainer/
RUN curl -L -o /usr/local/bin/pixi -fsSL --compressed "https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-$(uname -m)-unknown-linux-musl" \
    && chmod +x /usr/local/bin/pixi \
    && pixi info

ENV PIXI_HOME=/opt/pixi
RUN echo 'detached-environments = true' >/opt/pixi/config.toml

USER vscode
WORKDIR /home/vscode

RUN echo 'eval "$(pixi completion -s bash)"' >> /home/vscode/.bashrc
