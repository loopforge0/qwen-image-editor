# A bare Linux machine with nothing but git and curl, for testing the README's install steps from scratch.
# Built and run by run-fresh-test.ps1 / run-fresh-test.sh; see docker/README.md.
FROM ubuntu:24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Created here so named volumes (--keep-models) start out owned by the test user.
RUN useradd -m tester \
 && mkdir -p /home/tester/.local/share/qwen-image-editor/models /home/tester/.cache/uv \
 && chown -R tester:tester /home/tester

COPY fresh-test.sh /usr/local/bin/fresh-test
RUN chmod +x /usr/local/bin/fresh-test

USER tester
WORKDIR /home/tester
CMD ["fresh-test"]
