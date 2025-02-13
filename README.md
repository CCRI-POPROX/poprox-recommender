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

We manage software environments for this repository with [pixi][], and model and
data files with [dvc][]. The `pixi.lock` file provides a locked dependency set
for reproducibly running the recommender code with all dependencies on Linux and
macOS (we use the devcontainer for development support on Windows).

[pixi]: https://pixi.sh
[dvc]: https://dvc.org

The devcontainer automatically installs the development Pixi environment; if you
want to manually install it, you can run:

```console
pixi install -e dev
```

VS Code will also usually activate this environment by default when opening a
terminal; you can also directly run code in in the Pixi environment with any of
the following methods:

1.  Run a defined task, like `test`, with `pixi run`:

    ```console
    $ pixi run -e dev test
    ```

2.  Run individual commands with `pixi run`, e.g.:

    ```console
    $ pixi run -e dev pytest tests
    ```

3.  Run a Pixi shell, which activates the environment and adds the appropriate
    Python to your `PATH`:

    ```console
    $ pixi shell -e dev
    ```

> [!NOTE]
>
> If you have a CUDA-enabled Linux system, you can use the `dev-cuda` and
> `eval-cuda` environments to use your GPU for POPROX batch inference.

> [!NOTE]
>
> `pixi shell` starts a new, *nested* shell with the Pixi environment active. If
> you type `exit` in this shell, it will exit the nested shell and return you to
> you original shell session without the environment active.

## Data and Model Access

To get the data and models, there are two steps:

1.  Obtain the credentials for the S3 bucket and put them in `.env` (the environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`)
2.  `dvc pull`

## Local Endpoint Development

Local endpoint development also requires some Node-based tools in addition to the tools above — this install is
automated with `npm` and `pixi`:

```console
pixi run -e dev install-serverless
```

To run the API endpoint locally:

```console
pixi run -e dev start-serverless
```

<details>
<summary>Implementation</sumamry>

Under the hood, those tasks run the following Node commands within the `dev` environment:

```console
npm ci
npx serverless offline start --reloadHandler
```
</details>

Once the local server is running, you can send requests to `localhost:3000`. A request with this JSON body:

```json
{
  "interacted": [
    {
      "article_id": "e7605f12-a37a-4326-bf3c-3f9b72d0738d",
      "headline": "headline 1",
      "subhead": "subhead 1",
      "url": "url 1"
    },
    {
      "article_id": "a0266b75-2873-4a40-9373-4e216e88c2f7",
      "headline": "headline 2",
      "subhead": "subhead 2",
      "url": "url 2"
    },
    {
      "article_id": "d88ee3b6-2b5e-4821-98a8-ffd702f571de",
      "headline": "headline 3",
      "subhead": "subhead 3",
      "url": "url 3"
    }
  ],
  "candidates": [
    {
      "article_id": "7e5e0f12-d563-4a60-b90a-1737839389ff",
      "headline": "headline 4",
      "subhead": "subhead 4",
      "url": "url 4"
    },
    {
      "article_id": "7e5e0f12-d563-4a60-b90a-1737839389ff",
      "headline": "headline 5",
      "subhead": "subhead 5",
      "url": "url 5"
    },
    {
      "article_id": "2d5a25ba-0963-474a-8da6-5b312c87bb82",
      "headline": "headline 6",
      "subhead": "subhead 6",
      "url": "url 6"
    }
  ],
  "interest_profile": {
    "profile_id": "28838f05-23f5-4f23-bea2-30b51f67c538",
    "click_history": [
      {
        "article_id": "e7605f12-a37a-4326-bf3c-3f9b72d0738d"
      },
      {
        "article_id": "a0266b75-2873-4a40-9373-4e216e88c2f7"
      },
      {
        "article_id": "d88ee3b6-2b5e-4821-98a8-ffd702f571de"
      }
    ],
    "onboarding_topics": []
  },
  "num_recs": 1
}
```

should receive this response:

```json
{
    "recommendations": {
        "28838f05-23f5-4f23-bea2-30b51f67c538": [
            {
                "article_id": "7e5e0f12-d563-4a60-b90a-1737839389ff",
                "headline": "headline 5",
                "subhead": "subhead 5",
                "url": "url 5",
                "preview_image_id": null,
                "published_at": "1970-01-01T00:00:00Z",
                "mentions": [],
                "source": null,
                "external_id": null,
                "raw_data": null
            },
            ...
        ]
    },
    "recommender": {
        "name": "plain-NRMS",
        "version": null,
        "hash": "bd076520fa51dc70d8f74dfc9c0e0169236478d342b08f35b95399437f012563"
    }
}
```

You can test this by sending a request with curl:

```console
$ curl -X POST -H "Content-Type: application/json" -d @tests/request_data/basic-request.json localhost:3000
```


## Running the Evaluation

The default setup for this package is CPU-only, which works for basic testing
and for deployment, but is very inefficient for evaluation.  The current set of
models work on both CUDA (on Linux with NVidia cards) and MPS (macOS on Apple
Silicon).  To make use of a GPU, do the following:, run `dvc repro` under the
`eval-cuda` or `dev-cuda` environment (using either `pixi run` or `pixi shell`).

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
`pixi update`.  To update just `poprox-concepts`, run:

```console
pixi update poprox_concepts
```

To update all dependencies, run:

```console
pixi update
```

### Editor Setup

The devcontainer automatically configures several VS Code extensions and
settings; we also provide an `extensions.json` listing recommended extensions
for this repository.
