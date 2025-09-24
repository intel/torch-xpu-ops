# Docker images for GitHub CI and CD

This directory contains everything needed to build the Docker images
that are used in our CI tests.

## Docker CI builds

* `pytorch/manylinux2_28-builder:xpu-main` -- can use pytorch CICD image directly

## Docker CI tests

If also use this for build, need install Intel® Deep Learning Essentials,
refer to [Intel® Deep Learning Essentials](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=dl-essentials&dl-essentials-os=linux&dl-lin=offline)
```bash
# Build a specific image for tests
docker build --build-arg UBUNTU_VERSION=22.04 --file ubuntu/Dockerfile .
```
