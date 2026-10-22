# The View from Nowhere: Population-Centered Agent Evolution

This repository contains the code for Population-Centered Evolution (PCE).

## Requirements

- Linux (x86_64) with Docker
- Conda or Miniforge
- At least 500 GB of free disk for the benchmark images; 1 TB is recommended
- A DeepSeek API key
- Network access to PyPI, Hugging Face and GitHub during setup

## Setup

### 1. Host environment

```bash
conda env create -f environment.yml
conda activate mini_oracle
```

This installs Python 3.11. `requirements.txt` is installed inside the containers, not on the host.

### 2. Docker

```bash
docker run hello-world
```

On a permission error:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### 3. SWE-bench

Clone the evaluation harness into `swe_bench/SWE-bench`, apply the three patches, and install it
from that tree, not from PyPI:

```bash
cd swe_bench
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench
git checkout dc4c087c2b9e4cefebf2e3d201d27e362d899e0f
git apply ../../patches/0001-swebench-roman.patch
git apply ../../patches/0003-swebench-docutils-sphinx30.patch
git apply ../../patches/0004-swebench-list-images.patch
pip install -e .
cd ../../
```

### 4. Polyglot

`git` needs a `user.name` and a `user.email`, because the preparation step commits in each exercise
repository:

```bash
git config --global user.name "..."
git config --global user.email "..."
```

From the repository root, prepare the 225 exercises and build their images:

```bash
python -m polyglot.prepare_polyglot_dataset
bash polyglot/rebuild_images.sh
```

Re-running the first command changes every commit hash, so the second has to be re-run after it.

### 5. API key

Create `secrets.env` at the repository root:

```
DEEPSEEK_API_KEY=...
```

The solver is `deepseek-v4-flash`, and diagnosis and distillation use `deepseek-v4-pro`. DeepSeek has
since retired `deepseek-v4-flash`; the name is still accepted and is served by DeepSeek-V4.1-Flash.

## Running experiments

`exp/` holds the four experiments:

| Experiment | Benchmark | Method |
| --- | --- | --- |
| `gea` | SWE-bench | GEA (baseline) |
| `distill` | SWE-bench | PCE |
| `polyglot_gea` | Polyglot | GEA (baseline) |
| `polyglot_distill` | Polyglot | PCE |

Start one with:

```bash
./run_swe.sh gea
./run_swe.sh distill
./run_polyglot.sh polyglot_gea
./run_polyglot.sh polyglot_distill
```

To check an experiment without starting it, add `show`, for example `./run_swe.sh show distill`.

A SWE-bench experiment takes about four days and a Polyglot experiment about one day, depending on
the number of evaluation workers. Runs stay in the foreground, so start them in `tmux` or `screen`.
The first run builds the `dgm` image; add `--rebuild` after changing the code. The image name is
fixed, so only one checkout per machine can run at a time. To resume an interrupted run, add
`--continue-from output_dgm/<run>`. Results are written to `output_dgm/<run>/`.

## Acknowledgement

This codebase is built upon [GEA](https://github.com/UCSB-AI/GEA), which is built upon the
[Darwin Gödel Machine (DGM)](https://github.com/jennyzzt/dgm). We sincerely thank the authors for
their inspiring and impactful work.

The evaluation is based on the [SWE-bench](https://github.com/princeton-nlp/SWE-bench) and
[polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark) repositories.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
