# A bare Linux machine with the project already cloned and nothing else: no uv, no Python packages, no models.
# For walking through the README by hand. See docker/README.md.
#
#   docker build -t qie-manual -f docker/manual.Dockerfile .
#   docker run -it --rm --gpus all --ulimit memlock=-1 --shm-size 8g -p 127.0.0.1:7861:7861 qie-manual
FROM ubuntu:24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# The cache folders exist up front so named volumes mounted there start out owned by the test user.
RUN useradd -m tester \
 && mkdir -p /home/tester/.local/share/qwen-image-editor/models /home/tester/.cache/uv \
 && chown -R tester:tester /home/tester

COPY docker/update-code.sh /usr/local/bin/update-code
RUN chmod +x /usr/local/bin/update-code

# Stands in for `git clone`; .dockerignore leaves out what a clone wouldn't have (.venv, outputs, caches).
COPY --chown=tester:tester . /home/tester/qwen-image-editor

USER tester
WORKDIR /home/tester/qwen-image-editor
CMD ["bash", "-l"]
