"""
Workflow tasks scripted with Python Invoke.

.. note::

    This code needs to be able to run *without* the primary dependencies for
    ``poprox_recommender``.
"""

from __future__ import annotations

import re
from collections.abc import Generator
from os import fspath
from pathlib import Path
from tomllib import load as load_toml
from typing import Literal, TypeAlias

from dvc.api import DVCFileSystem
from invoke.context import Context
from invoke.tasks import task
from wcmatch import glob

FileAccess: TypeAlias = Literal["public", "shared", "private"]

REPO_DIR = Path(__file__).parent


@task
def list_artifacts(c: Context, public=False, shared=False):
    "List DVC-tracked files."
    for file, access in _scan_dvc_files():
        if public and access == "public":
            print(file)
        elif shared and access != "private":
            print(file)
        elif not shared and not public:
            print(f"{access}\t{file}")


@task
def upload_shared_data(c: Context, public=False, dry_run=False, verbose=False):
    "Upload shared or public data."
    target = "public" if public else "shared"
    up_files = []
    up_dirs = []
    for file, access in _scan_dvc_files():
        if access == "public" or (access == "shared" and not public):
            if file[-1] == "/":
                up_dirs.append(file)
            else:
                up_files.append(file)

    for dir in up_dirs:
        print(f"uploading {target} directory {dir}")
        if not dry_run:
            c.run(f"dvc push -r {target} --no-run-cache -R {dir}")

    # filter out files covered by directories
    up_files = [f for f in up_files if not any(f.startswith(d) for d in up_dirs)]
    if verbose:
        for file in up_files:
            print(f"uploading {target} file {file}")
    else:
        print(f"uploading {len(up_files)} {target} files")
    if not dry_run:
        c.run(f"dvc push -r {target} --no-run-cache " + " ".join(up_files))


class SharedAccessCheck:
    public_inc: list[re.Pattern]
    shared_inc: list[re.Pattern]

    def __init__(self):
        ds_f = REPO_DIR / "data-sharing.toml"

        with ds_f.open("rb") as inf:
            sharing = load_toml(inf)

        # parse public patterns
        inc, exc = glob.translate(sharing["data"]["public"]["patterns"])
        self.public_inc = [re.compile(p) for p in inc]

        # parse shared patterns
        inc, exc = glob.translate(sharing["data"]["shared"]["patterns"])
        self.shared_inc = [re.compile(p) for p in inc]

    def access(self, path) -> FileAccess:
        for pattern in self.public_inc:
            if pattern.match(path):
                return "public"

        for pattern in self.shared_inc:
            if pattern.match(path):
                return "shared"

        return "private"


def _scan_dvc_files() -> Generator[tuple[str, FileAccess], None, None]:
    sam = SharedAccessCheck()
    fs = DVCFileSystem(fspath(REPO_DIR))
    files = fs.find("/", withdirs=True, dvc_only=True)
    for file in files:
        # deal with leading /
        assert file[0] == "/"
        file = file[1:]
        path = REPO_DIR / file

        if path.is_dir():
            file = file + "/"

        access = sam.access(file)
        yield file, access
