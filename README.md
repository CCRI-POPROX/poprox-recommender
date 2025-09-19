# poprox-recommender

This repository contains the POPROX recommender code — the end-to-end logic for
producing recommendations using article data, user histories and profiles, and
trained models.

- [Installation](#installation)
- [Local Development](#localdevelopment)
- [Editor Setup](#editor-setup)
- [License](#license)

## Installation for Development

This repository includes a devcontainer configuration that we recommend using
for development and testing of the recommender code.  It is not a good solution
for running evaluations (see below for non-container setup), but is the easiest
and most reliable way to set up your development environment across platforms.

To use the devcontainer, you need:

- VS Code (other editors supporting DevContainer may also work, but this is the
  best-supported and best-tested).
- Docker (probably also works with Podman or other container CLIs, but we test
  with Docker).

With those installed, open the repository in VS Code, and it should prompt you
to re-open in the dev container; if it does not, open the command palette
(<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>) and choose “Dev Containers:
Rebuild and Reopen in Container”.

### Installing Docker

On Linux, install the [Docker Engine][engine], and add your user to the `docker`
group so you can create containers without root.

On Windows, install [Docker Desktop][dd], [Rancher Desktop][rancher], or
similar.

On MacOS, you can install Docker or Rancher Desktop linked above, or you can use
[Colima][], which we recommend for simplicity and licensing clarity.  To install
and use Colima:

```console
$ brew install colima docker
$ colima start -m 4
```

It should also be possible to directly use Lima, but we have not tested or
documented support for that.

[engine]: https://docs.docker.com/engine/install/
[dd]: https://www.docker.com/products/docker-desktop/
[rancher]: https://rancherdesktop.io/
[Colima]: https://github.com/abiosoft/colima

## Working with the Software

We manage software environments for this repository with [uv][], and model and
data files with [dvc][]. The `uv.lock` file provides a locked dependency set
for reproducibly running the recommender code with all dependencies on Linux and
macOS (we use the devcontainer for development support on Windows).

[uv]: https://docs.astral.sh/uv
[uv-inst]: https://docs.astral.sh/uv/getting-started/installation/
[dvc]: https://dvc.org

The devcontainer automatically installs the development environment. If you want to manually
install the software, first install `uv` (see the [install instructions][uv-inst]), then run:

```console
$ uv sync --group cpu
Resolved 304 packages in 10ms
      Built poprox-recommender @ file:///Users/mde48/POPROX/poprox-recommender
      Built poprox-concepts @ git+https://github.com/CCRI-POPROX/poprox-concepts.git@d0e27c90f6eddcc4f041e5d871ff4a13c2ec70f7
      Built antlr4-python3-runtime==4.9.3
      Built pylatex==1.4.2
Prepared 223 packages in 19.45s
Installed 278 packages in 1.80s
```

> [!NOTE]
>
> If you have a CUDA-enabled Linux system, you can use the `cuda` extra to get
> CUDA-enabled PyTorch for POPROX batch inference and model training.  To
> install this, run:
>
> ```console
> $ uv sync --group cuda
> ```

The dev container also automatically activates the environment in its terminal.
To activate manually, run the following in each shell session:

```console
$ . .venv/bin/activate
```

You can also run things from inside the environment with `uv run`:

```console
$ uv run dvc pull
```

A few useful commands for the terminal:

1.  Run the tests:

    ```console
    $ pytest tests
    ```

## Data and Model Access

To get the data and models, there are two steps:

1.  Obtain the credentials for the S3 bucket and put them in `.env` (the environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`)
2.  `dvc pull`

## Local Endpoint Development

Local endpoint testing requires building and running the Docker image:

```console
$ docker buildx build -t poprox-recommender:test .
$ docker run -d -p 9000:8080 --name=recommender poprox-recommender:test
```

You can then send a request to the endpoint:

```console
$ python scripts/send-request.py -p 9000
```

Pass the `-h` option to `send-request.py` to see command-line options.

## Running the Evaluation

The default setup for this package is CPU-only, which works for basic testing
and for deployment, but is very inefficient for evaluation.  The current set of
models work on both CUDA (on Linux with NVidia cards) and MPS (macOS on Apple
Silicon).  To make use of a GPU, install with the `cuda` extra (and the `eval` group,
which is included by default).  Run the evaluation with `dvc`:

```console
$ dvc repro measure-mind-small
```

You can also run `measure-mind-val` or `measure-mind-subset`.

### Evaluation Timing

Timing information for generating recommendations with the MIND validation set:

| Machine      | CPU                | GPU         | Rec. Time | Rec. Power | Eval Time |
| ------------ | :----------------: | :---------: | :-------: | :--------: | :-------: |
| [Cruncher][] | EPYC 7662 (2GHz)   | A40 (CUDA)  | 45m¹      | 418.5 Wh   | 24m       |
| [Screamer][] | i9 14900K (3.2GHz) | 4090 (CUDA) | 28m16s²   |            | 14m       |
| [Ranger][]   | Apple M2 Pro       | -           | <20hr³    |            | 30m³      |
| [Ranger][]   | Apple M2 Pro       | M2 (MPS)    | <12hr³    |            |           |

[Cruncher]: https://codex.lenskit.org/hardware/cruncher.html
[Screamer]: https://codex.lenskit.org/hardware/screamer.html
[Ranger]: https://codex.lenskit.org/hardware/ranger.html

Footnotes:

1. Using 12 worker processes
2. Using 8 worker processes
3. Estimated based on early progress, not run to completion.

## Training Models

Model training is also orchestrated by DVC, along with its preprocessing stages.
To avoid accidentally retraining models, the model training stages are *frozen*,
and we un-freeze them when we need to re-train.

To re-train the model, run:

```console
$ dvc pull
$ dvc unfreeze models/dvc.yaml:train-nrms-mind
Modifying stage 'train-nrms-mind' in 'models/dvc.yaml'
$ dvc repro models/dvc.yaml:train-nrms-mind
$ dvc freeze models/dvc.yaml:train-nrms-mind
Modifying stage 'train-nrms-mind' in 'models/dvc.yaml'
$ git add models
$ git commit
$ dvc push
$ git push
```

## Additional Software Environment Notes

### Non-Container Development Notes

If you are not using the devcontainer, set up `pre-commit` to make sure that
code formatting rules are applied as you make changes:

```console
pre-commit install
```

### Dependency Updates

If you update the dependencies in poprox-recommender, or add code that requires
a newer version of `poprox-concepts`, you need to regenerate the lock file with
`uv sync`.  To update just `poprox-concepts`, run:

```console
$ uv sync -P poprox-concepts
```

To update all dependencies, run:

```console
$ uv sync -U
```

### Editor Setup

The devcontainer automatically configures several VS Code extensions and
settings; we also provide an `extensions.json` listing recommended extensions
for this repository.
