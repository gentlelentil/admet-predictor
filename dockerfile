FROM continuumio/miniconda3:latest

# environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /admet-predictor

COPY environment.yaml /admet-predictor/

#create conda environment and remove cache
RUN conda env create -f environment.yaml && \
    conda clean --all --yes

SHELL ["conda", "run", "-n", "rdkitenv", "/bin/bash", "-c"]

COPY . /admet-predictor/

EXPOSE 5001

CMD ["conda", "run", "--no-capture-output", "-n", "rdkitenv", "gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "ml_api:app"]

