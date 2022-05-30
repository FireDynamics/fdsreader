# Dockerfile for continuous integration (github pages and pypi deployment)
FROM travisci/ci-sardonyx:packer-1652254210-b649fb09

SHELL ["/bin/bash", "-c"]

ENV PYPI_TOKEN=""
ENV GITHUB_TOKEN=""

COPY docker_deployment.sh /docker_deployment.sh
ENTRYPOINT bash /docker_deployment.sh