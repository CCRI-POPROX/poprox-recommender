FROM mcr.microsoft.com/devcontainers/base:bookworm
ARG PIXI_VERSION=v0.40.3

# derived from https://pixi.sh/latest/ide_integration/devcontainer/
RUN curl -L -o /usr/local/bin/pixi -fsSL --compressed "https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-$(uname -m)-unknown-linux-musl" \
    && chmod +x /usr/local/bin/pixi \
    && pixi info

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN echo 'eval "$(pixi completion -s bash)"' >> /home/vscode/.bashrc
