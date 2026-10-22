# The View from Nowhere: Population-Centered Agent Evolution

Code for Population-Centered Evolution (PCE) and for GEA, the baseline it is compared against. The
code is built on [GEA](https://github.com/UCSB-AI/GEA).

## Requirements

- Linux, about 300 GB of free disk for both benchmarks' images, and Conda or Miniforge.
- Network access for the model API, PyPI, Hugging Face and GitHub during setup.

## 1. Docker

```bash
docker run hello-world
```

On a permission error:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## 2. Host environment

```bash
conda env create -f environment.yml
conda activate mini_oracle
```

Python 3.11. `requirements.txt` is a different list: it is installed *inside* the containers by
`Dockerfile` and by the two harnesses, and is not the host environment.

## 3. SWE-bench

Vendor the evaluation harness into `swe_bench/SWE-bench`, apply the three patches, and install it
from that tree:

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

Install from this tree, not from PyPI, which ships the harness unpatched. 

The instance data is `princeton-nlp/SWE-bench_Verified`, downloaded from Hugging Face on first
use. The graded subset is `swe_bench/subsets/task.json`.

## 4. Polyglot

`git` must have a `user.name` and a `user.email`, because the preparation step commits in each
exercise repository:

```bash
git config --global user.name "..."
git config --global user.email "..."
```

The first command below clones `Aider-AI/polyglot-benchmark`, rewrites each of the 225 exercises
into a two-commit git repository, and writes `polyglot/polyglot_benchmark_metadata.json`. The
second builds the 225 instance images. Run both from the repository root.

```bash
python -m polyglot.prepare_polyglot_dataset
bash polyglot/rebuild_images.sh
```

Re-running the first changes every commit hash, so the second has to be re-run after it. The
graded subset is `polyglot/subsets/task.json`.

## 5. Docker images

Three kinds, all built locally:

| Image | Built by | When |
| --- | --- | --- |
| `dgm` | `run.py`, from `Dockerfile` | first launch, or `--rebuild` |
| `sweb.base`, `sweb.env.*`, `sweb.eval.*` | the SWE-bench harness | on demand, per instance |
| `pb.base`, `pb.env.*`, `pb.eval.*` | `polyglot/rebuild_images.sh` | step 4 |

The `dgm` image name is fixed, so one checkout per machine can be live at a time. `run.py`
compares the image against the working tree and refuses to launch when they differ.

## 6. Credentials

`secrets.env` at the repository root, read by the launch scripts:

```
DEEPSEEK_API_KEY=...
```

`ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are passed into the containers when set and are not
reached by any of the four arms below.

The arms run `dsclaude/deepseek-v4-flash` as the solver, against
`https://api.deepseek.com/anthropic`, and `deepseek-v4-pro` for the diagnosis and the
distillation, against `https://api.deepseek.com`. For a different solver endpoint set
`DEEPSEEK_ANTHROPIC_BASE_URL` and give `DEEPSEEK_API_KEY` that host's key.

DeepSeek has since retired `deepseek-v4-flash`; the name is still accepted and is served by
DeepSeek-V4.1-Flash.

## 7. Arms

Each arm is one file under `exp/`. `run.py` resolves it to an environment and a `GEA_outer.py`
command line, and writes `exp/<ID>.<stamp>.launch.txt` recording what the run received.

`gea` and `polyglot_gea` run GEA; `distill` and `polyglot_distill` run PCE.

| Arm | Benchmark | Relative to the baseline |
| --- | --- | --- |
| `gea` | SWE-bench | baseline |
| `distill` | SWE-bench | `DISTILL=1`, `DISTILL_STEPS=250`, `EMPTY_ENTRIES=1` |
| `polyglot_gea` | Polyglot | baseline |
| `polyglot_distill` | Polyglot | `DISTILL=1`, `DISTILL_STEPS=250`, `EVAL_MAX_WORKERS=4` (12 in the baseline) |

## 8. Launch

```bash
./run_swe.sh distill
./run_polyglot.sh polyglot_distill
```

Resolve an arm and print it without launching:

```bash
./run_swe.sh show distill
```

List every arm:

```bash
python run.py --list
```

Runs are foreground and take about a day. Add `--rebuild` when the working tree has moved since
the `dgm` image was built, and `--continue-from output_dgm/<stamp>` to resume a run.

## 9. Output

`output_dgm/<stamp>/` holds `dgm_metadata.jsonl`, one directory per child, and `_agent_src`, the
source frozen at the start of the run. Each child directory holds `metadata.json` with that
child's score, `model_patch.diff`, `self_evo.md`, and `predictions/`. Distillations are written to
`_distill/<tag>/verdict_<tag>.json`.

## Acknowledgement

This codebase is built upon [GEA](https://github.com/UCSB-AI/GEA), which is built upon the
[Darwin Gödel Machine (DGM)](https://github.com/jennyzzt/dgm). We sincerely thank the authors for
their inspiring and impactful work.

The evaluation is based on the [SWE-bench](https://github.com/princeton-nlp/SWE-bench) and
[polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark) repositories.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
