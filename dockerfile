FROM continuumio/miniconda3:latest

# environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY environment.yml /app/

#create conda environment and remove cache
RUN conda env create -f environment.yml && \
    conda clean --all --yes

SHELL ["conda", "run", "-n", "rdkit_datasets", "/bin/bash", "-c"]

COPY . /app/

# run the application
ENTRYPOINT ["conda", "run", "-n", "rdkit_datasets", "python", "ML_pipeline.py"] 