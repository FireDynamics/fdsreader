# Dockerfile for continuous integration (github pages and pypi deployment)
FROM travisci/ci-sardonyx:packer-1652254210-b649fb09

SHELL ["/bin/bash", "-c"]

ENV PYPI_TOKEN="pypi-AgEIcHlwaS5vcmcCJDhmZGU5MTA4LWNmNTAtNGU3OC04YmVlLWM3MTcyM2VhOGQ0MwACOnsicGVybWlzc2lvbnMiOiB7InByb2plY3RzIjogWyJmZHNyZWFkZXIiXX0sICJ2ZXJzaW9uIjogMX0AAAYgaKbLGoF3ycxd05of8KPaOKVE1QfEPLpnAyoFIo6kUwg"
ENV GITHUB_TOKEN="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"

COPY docker_deployment.sh /docker_deployment.sh
ENTRYPOINT bash /docker_deployment.sh