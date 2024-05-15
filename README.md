# poprox-recommender

This repository contains the POPROX recommender code — the end-to-end logic for
producing recommendations using article data, user histories and profiles, and
trained models.

Model and data files are managed using [dvc][].  The `conda-lock.yml` provides a
[conda-lock][] lockfile for reproducibly creating an environment with all
necessary dependencies.

[dvc]: https://dvc.org
[conda-lock]: https://conda.github.io/conda-lock/

To set up the environment with Conda:

    conda install -n base -c conda-forge conda-lock
    conda lock install -n poprox-recsys
    conda activate poprox-recsys

If you use `micromamba` instead of a full Conda installation, it can directly use the lockfile:

    micromamba create -n poprox-recs -f conda-lock.yml

To get the data and models, there are two steps:

1.  Obtain the credentials for the S3 bucket and put them in `.env` (the environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`)
2.  `dvc pull`