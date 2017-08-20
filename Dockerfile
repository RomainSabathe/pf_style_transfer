FROM python:2-onbuild

# Inspired by https://github.com/prakhar1989/docker-curriculum
MAINTAINER https://github.com/RomainSabathe

EXPOSE 5000
CMD ["python", "./run.py"]
