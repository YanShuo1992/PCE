from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "exp"

META_KEYS = {"ID", "SUMMARY", "STATUS", "WHY", "CRITERION"}

CLI_KEYS = {
    "MAX_GENERATION": "--max_generation",
    "SELFIMPROVE_SIZE": "--selfimprove_size",
    "SELFIMPROVE_WORKERS": "--selfimprove_workers",
    "GROUPIMPROVE_SIZE": "--groupimprove_size",
    "CHOOSE_SELFIMPROVES_METHOD": "--choose_selfimproves_method",
    "UPDATE_ARCHIVE": "--update_archive",
    "DIAGNOSE_MODEL": "--diagnose_model",
    "NUM_SWE_EVALS": "--num_swe_evals",
    "EVAL_NOISE": "--eval_noise",
    "CONTINUE_FROM": "--continue_from",
    "RUN_BASELINE": "--run_baseline",
}
FLAG_KEYS = {
    "NO_FULL_EVAL": "--no_full_eval",
    "POLYGLOT": "--polyglot",
    "SHALLOW_EVAL": "--shallow_eval",
}
ENV_KEYS = {
    "CODING_AGENT_CLAUDE_MODEL",
    "CODING_MAX_TOKENS",
    "EVAL_MAX_WORKERS",
    "DIAGNOSE_REASONING_EFFORT",
    "DEEPSEEK_THINKING",
    "SWE_SOLVE_TIMEOUT",
    "POLYGLOT_SOLVE_TIMEOUT",
    "FORBID_ANSWER_LOOKUP",
    "ALLOW_AGENT_NETWORK",
    "EMPTY_ENTRIES",
    "POLYGLOT_ENTRY_POOL",
    "SELFIMPROVE_MAX_TOKENS",
    "SELFIMPROVE_TRUNCATION_RETRIES",
    "CONTRAST",
    "DISTILL",
    "DISTILL_MODEL",
    "DISTILL_STEPS",
}

IMAGE_FILES = ["coding_agent.py", "coding_agent_polyglot.py", "llm.py", "llm_withtools.py",
               "harness_def.py", "model_io.py",
               "requirements.txt"]
IMAGE_DIRS = ["tools", "utils", "prompts"]
IMAGE_NAME = "dgm"

def _dockerignored() -> set[str]:
    f = ROOT / ".dockerignore"
    if not f.exists():
        return set()
    out = set()
    for line in f.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "<!--")) or s.endswith("/") or "*" in s:
            continue
        out.add(s)
    return out

def _agent_side_files() -> list[str]:
    skip = _dockerignored()
    out = [f for f in IMAGE_FILES if (ROOT / f).exists() and f not in skip]
    for d in IMAGE_DIRS:
        out += sorted(str(f.relative_to(ROOT)) for f in (ROOT / d).glob("*.py")
                      if str(f.relative_to(ROOT)) not in skip)
    return out

def check_tree_complete() -> bool:
    import agent_src
    missing = ([n for n in agent_src.FILES if not (agent_src.ROOT / n).is_file()] +
               [n + "/" for n in agent_src.DIRS if not (agent_src.ROOT / n).is_dir()])
    if missing:
        print(f"X the agent source tree is incomplete -- {len(missing)} declared entr"
              f"{'y is' if len(missing) == 1 else 'ies are'} missing from {agent_src.ROOT}:")
        for m in missing:
            print(f"    {m}")
        print("  Nothing is launched. Restore them (git checkout <commit> -- <path>) and retry.")
        return False
    print(f"  \u2713 agent source tree complete "
          f"({len(agent_src.FILES)} files, {len(agent_src.DIRS)} directories)")
    return True

def check_image_fresh(dry: bool = False, exp_id: str = "<arm>", rebuilding: bool = False) -> bool:
    import hashlib
    have = subprocess.run(
        ["docker", "images", "-q", IMAGE_NAME], capture_output=True, text=True)
    if not have.stdout.strip():
        if dry:
            print(f"  image {IMAGE_NAME} does not exist -> a real run would build it here, serially")
            return True
        print(f"  image {IMAGE_NAME} does not exist -> building it once, serially "
              f"(GEA would have {IMAGE_NAME}-raced across workers)")
        r = subprocess.run(["docker", "build", "-t", IMAGE_NAME, "."],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print("X docker build failed:")
            print("\n".join(r.stderr.splitlines()[-15:]))
            return False
        print(f"  OK built {IMAGE_NAME}")
        return True

    files = _agent_side_files()
    r = subprocess.run(
        ["docker", "run", "--rm", IMAGE_NAME, "md5sum", *[f"/dgm/{f}" for f in files]],
        capture_output=True, text=True)
    in_image = {}
    for line in r.stdout.splitlines():
        h, _, path = line.partition("  ")
        if path.startswith("/dgm/"):
            in_image[path[len("/dgm/"):]] = h.strip()

    stale, missing = [], []
    for f in files:
        local = hashlib.md5((ROOT / f).read_bytes()).hexdigest()
        got = in_image.get(f)
        if got is None:
            missing.append(f)
        elif got != local:
            stale.append(f)

    if not stale and not missing:
        print(f"  ✓ image {IMAGE_NAME} matches the working tree ({len(files)} agent-side files)")
        return True
    if rebuilding:
        print(f"  image {IMAGE_NAME} is stale, which --rebuild fixes: "
              f"{', '.join(stale + missing)}")
        return True
    print(f"✗ image {IMAGE_NAME} is **stale** -- the coding agent would run old code, "
          f"and the log would only say Skipping build.")
    for f in stale:
        print(f"    content differs: {f}")
    for f in missing:
        print(f"    not in the image: {f}")
    print(f"\n  How to fix (pick one):")
    print(f"    python run.py {exp_id} --rebuild     # delete the image and let it rebuild itself")
    print(f"    docker rmi {IMAGE_NAME}")
    return False

def read_conf(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: the top level has to be a mapping of KEY: value")
    conf = {}
    for k, v in raw.items():
        if v is None:
            conf[str(k)] = ""
        elif isinstance(v, bool):
            conf[str(k)] = "1" if v else "0"
        elif isinstance(v, (str, int, float)):
            conf[str(k)] = str(v)
        else:
            raise SystemExit(f"{path}: {k} holds a {type(v).__name__}; only single scalars are "
                             f"allowed, one key per line")
    return conf

def build(conf: dict) -> tuple[list[str], dict]:
    unknown = set(conf) - META_KEYS - set(CLI_KEYS) - set(FLAG_KEYS) - ENV_KEYS
    if unknown:
        raise SystemExit(
            f"✗ unrecognized config keys: {sorted(unknown)}\n"
            f"  available: {sorted(set(CLI_KEYS) | set(FLAG_KEYS) | ENV_KEYS)}\n"
            f"  (to add a new key, add it to the tables in run.py first; "
            f"do not count on it being ignored)")

    cmd = [sys.executable, "GEA_outer.py"]
    for key, flag in CLI_KEYS.items():
        v = conf.get(key, "")
        if v:
            cmd += [flag, v]
    FALSEY = {"", "0", "false", "no", "off", "none"}
    for key, flag in FLAG_KEYS.items():
        v = conf.get(key, "").strip().lower()
        if v in FALSEY:
            continue
        if v not in {"1", "true", "yes", "on"}:
            raise SystemExit(f"X {key}={conf[key]!r} is not a recognised on/off value. "
                             f"Use 1/true/yes/on or 0/false/no/off (empty = off).")
        cmd.append(flag)

    env = {k: conf[k] for k in ENV_KEYS if conf.get(k, "")}
    return cmd, env

def list_exps() -> None:
    rows = []
    for f in sorted(EXP.glob("*.yaml")):
        c = read_conf(f)
        rows.append((c.get("ID", f.stem), c.get("STATUS", "?"), c.get("SUMMARY", "")))
    if not rows:
        print(f"no .yaml under {EXP}")
        return
    w = max(len(r[0]) for r in rows)
    for i, status, summary in rows:
        print(f"  {i:<{w}}  {status:<8}  {summary}")

def _fingerprint(secret: str | None) -> str:
    if not secret:
        return "<unset>"
    import hashlib
    return hashlib.sha256(secret.encode()).hexdigest()[:12]

def main() -> int:
    ap = argparse.ArgumentParser(description="Launch GEA from exp/<ID>.yaml")
    ap.add_argument("exp_id", nargs="?", help="experiment id, corresponds to exp/<ID>.yaml")
    ap.add_argument("--list", action="store_true", help="list every experiment")
    ap.add_argument("--dry", action="store_true",
                    help="only print how it would be launched, do not actually run")
    ap.add_argument("--continue-from", "--continue_from", dest="continue_from", metavar="RUN_DIR",
                    help="resume an existing run directory instead of starting a new one. "
                         "GEA_outer reads that run's dgm_metadata.jsonl, takes its archive and "
                         "starts at the generation after the last recorded one, and reuses the "
                         "_agent_src snapshot rather than the working tree. Given here rather "
                         "than in the arm file because it names one run, not a configuration -- "
                         "and because an arm file that carries it would silently resume that "
                         "same run the next time anyone launched the arm.")
    ap.add_argument("--rebuild", action="store_true",
                    help="delete the dgm image first to force a rebuild "
                         "(use this after changing agent-side code)")
    a = ap.parse_args()

    if a.list or not a.exp_id:
        list_exps()
        return 0

    path = EXP / f"{a.exp_id}.yaml"
    if not path.exists():
        print(f"✗ no such file: {path}")
        list_exps()
        return 1

    conf = read_conf(path)
    if a.continue_from:
        prev = Path(a.continue_from)
        if not (prev / "dgm_metadata.jsonl").is_file():
            print(f"X --continue-from: {prev} has no dgm_metadata.jsonl, so there is nothing to")
            print(f"  resume. Give the run directory itself, e.g. output_dgm/<stamp>_<pid>.")
            return 1
        conf["CONTINUE_FROM"] = str(prev)
    cmd, env_add = build(conf)

    print("=" * 70)
    print(f"experiment {conf.get('ID', a.exp_id)}   status {conf.get('STATUS', '?')}")
    print(f"  {conf.get('SUMMARY', '')}")
    if conf.get("WHY"):
        print(f"  why:       {conf['WHY']}")
    if conf.get("CRITERION"):
        print(f"  criterion: {conf['CRITERION']}")
    print("=" * 70)
    print("environment variables:")
    for k, v in sorted(env_add.items()):
        print(f"  {k}={v}")
    print("command:")
    print("  " + " ".join(cmd))
    print("=" * 70)

    if conf.get("STATUS") == "dead":
        print("✗ this experiment is marked STATUS=dead, not running it.")
        return 1

    missing = [k for k in ("DEEPSEEK_API_KEY",) if not os.getenv(k)]
    if missing:
        print(f"✗ missing environment variables: {missing}")
        print("   set -a; . secrets.env; set +a")
        return 2
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
    except Exception:
        print("✗ docker is not available")
        return 2

    if a.dry:
        import net_policy as _np
        print(f"  network: {len(_np.BLOCKED_HOSTS)} host(s) would be blocked in every container"
              if _np.enabled() else
              f"  network: UNRESTRICTED -- {_np.ENV} is set, the agent could fetch the upstream fix")
        if a.rebuild:
            print(f"  --dry: NOT deleting image {IMAGE_NAME}; a real --rebuild would remove it "
                  f"and build it once, serially, before launching")
        check_tree_complete()
        check_image_fresh(dry=True, exp_id=a.exp_id, rebuilding=a.rebuild)
        print("(--dry: stopping here, nothing was actually run)")
        return 0

    if a.rebuild:
        try:
            _live = subprocess.run(["pgrep", "-f", "GEA_outer.py --max_generation"],
                                   capture_output=True, text=True).stdout.split()
        except Exception:
            _live = []
        if _live:
            print(f"X --rebuild refused: {len(_live)} evolution run(s) are live (pid "
                  f"{', '.join(_live)}).")
            print("  The image is the source those runs' children edit. Rebuilding now changes it")
            print("  under them from the next generation on, and nothing would say so afterwards.")
            print("  Wait for them to finish, or launch without --rebuild if the image already")
            print("  matches what this experiment needs.")
            return 4
        print(f"  --rebuild: deleting image {IMAGE_NAME} so it gets rebuilt")
        subprocess.run(["docker", "rmi", "-f", IMAGE_NAME], capture_output=True)
    if not check_tree_complete():
        return 3
    if not check_image_fresh(exp_id=a.exp_id, rebuilding=a.rebuild):
        return 3

    import net_policy
    if net_policy.install():
        print(f"  network: {len(net_policy.BLOCKED_HOSTS)} host(s) blocked in every container "
              f"(package indexes, code hosts, search). {net_policy.ENV}=1 turns this off.")
    else:
        print(f"  network: UNRESTRICTED -- {net_policy.ENV} is set. The agent can fetch the "
              f"upstream fix, and one has.")

    env = {**os.environ, **env_add}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    started = datetime.now().astimezone()
    started_line = (f"started={started.strftime('%Y-%m-%d %H:%M:%S %z')} ({started.tzname()})"
                    f"  utc={started.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n")
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip() or "<unknown>"
        names = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
    except Exception:
        head, names = "<unknown>", ""
    log = EXP / f"{a.exp_id}.{stamp}.launch.txt"
    log.write_text(
        f"exp={a.exp_id}\nstamp={stamp}\n" + started_line
        + f"cwd={ROOT}\ngit={head}{(' ' + names) if names else ''}\n"
        + "".join(f"env {k}={env.get(k, '<unset>')}\n"
                  for k in sorted(ENV_KEYS | set(env_add)))
        + "".join(f"key {k}={_fingerprint(env.get(k))}\n"
                  for k in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
                  if env.get(k))
        + "cmd " + " ".join(cmd) + "\n"
        + "".join(f"conf {k}={v}\n" for k, v in sorted(conf.items())))
    print(f"launch record: {log}")
    print("Running in the foreground, no nohup -- it has to stay visible.\n")
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode

if __name__ == "__main__":
    sys.exit(main())
