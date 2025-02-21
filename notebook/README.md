# MOE Scaling law - notebook

A python script and notebook for demonstrating the proposed
MOE scaling law -
[see `README.md`](https://github.com/wrmedford/moe-scaling))
by Wes Medford, et al.

---

## Requirements

* `uv` (or a supported python version/manager)
* `just` for running `justfile` commands

## Run the notebook

Use the `justfile` commands to bootstrap
a new venv, get the dependencies,
and start the notebook:

```justfile
# bootstrap a uv virtual env
venv:
  uv venv

# grab and sync all the deps in the pyproject via uv
sync-deps:
  uv sync

# run the Jupyter notebook through uv
run:
  uv run --with jupyter jupyter lab

# format will format the py code using ruff
format:
  uv run ruff format ./

# WARNING: gets rid of checkpoints and existing notebook contents
clean-notebook:
  uv run --with jupyter jupyter nbconvert --clear-output --inplace moe-scaling-law.ipynb

# "EVERYTHING."
all: venv sync-deps run
```
