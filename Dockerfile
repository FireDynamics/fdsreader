# Dockerfile for continuous integration (github pages and pypi deployment)
FROM travisci/ci-sardonyx:packer-1652254210-b649fb09

SHELL ["/bin/bash", "-c"]

ENV PYPI_TOKEN="pypi-AgEIcHlwaS5vcmcCJDc1ZTJjMDExLTQ2MjgtNDQzNy1hOWI5LWJiNGE5NmQyMzM1NgACOnsicGVybWlzc2lvbnMiOiB7InByb2plY3RzIjogWyJmZHNyZWFkZXIiXX0sICJ2ZXJzaW9uIjogMX0AAAYg2w070JRiMS6hFml91Za8POFR0za1k5xpbjzKHiN-74Q"
ENV GITHUB_TOKEN="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"

COPY docker_deployment.sh /docker_deployment.sh
ENTRYPOINT bash /docker_deployment.sh