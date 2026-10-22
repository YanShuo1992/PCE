from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_MEMBERS = 3

SCRATCH_DIR = "_distill_scratch"
OUT_DIR = "_distill_out"
sys.path.insert(0, str(ROOT))

OBSERVATION_TEMPLATE = (
    "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
    "<returncode>{{output.returncode}}</returncode>\n"
    "{% if output.output | length < 10000 -%}\n<output>\n{{ output.output -}}\n</output>\n"
    "{%- else -%}\n<warning>\nThe output of your last command was too long. "
    "Use head, tail, sed or a more selective pattern.\n</warning>\n"
    "{%- set elided = output.output | length - 10000 -%}\n"
    "<output_head>\n{{ output.output[:5000] }}\n</output_head>\n"
    "<elided_chars>\n{{ elided }} characters elided\n</elided_chars>\n"
    "<output_tail>\n{{ output.output[-5000:] }}\n</output_tail>\n{%- endif -%}"
)

SYNC_EXCLUDES = ["--exclude=*_eval.sh", "--exclude=eval.sh"]

def purge_excluded(work) -> int:
    from pathlib import Path as _P
    n = 0
    for pat in ("**/*_eval.sh", "**/eval.sh"):
        for f in _P(work).glob(pat):
            try:
                f.unlink(); n += 1
            except OSError:
                pass
    return n

def make_tracing_agent(base, verbose: bool):
    class TracingAgent(base):
        def query(self):
            m = super().query()
            if not verbose:
                return m
            try:
                for act in (m.get("extra") or {}).get("actions", []):
                    cmd = str(act.get("command", act)).replace("\n", " ⏎ ")
                    print(f"    [{self.n_calls:3}] $ {cmd[:150]}", flush=True)
            except Exception:
                pass
            return m

        def execute_actions(self, message):
            obs = super().execute_actions(message)
            if not verbose:
                return obs
            try:
                for o in obs:
                    txt = str(o.get("content"))
                    rc = "returncode>" in txt and txt.split("returncode>")[1].split("<")[0] or "?"
                    print(f"          -> rc={rc}  {len(txt)} chars", flush=True)
            except Exception:
                pass
            return obs

    return TracingAgent

SYSTEM = """\
The population you are looking at is evolving a HARNESS. The model that writes the code is fixed
and cannot be changed. The harness is everything around it: the instructions it is handed before it
starts, the tools it can call, what it is told to do after an edit and after a failure, and what
the loop does when a reply comes back unusable. Every agent in the population is the same model
behind a different harness.

Your job is to find ONE harness-level evolution point: something about the harness that the
population's behaviour on a single instance shows to be missing, absent, or wrong. A later step
turns that into a concrete change; you do not write the change and you do not fix the issue.

You have bash. Everything under your working directory is a copy; nothing you do can affect the run
it came from. Stay inside it.
"""

INSTANCE = """\
# The instance

`{instance}`

Every agent directory in your working directory is one member of the population: the same model,
a different harness, all attempted this instance. For any agent `<commit>`:

    <commit>/metadata.json                                        its own overall score, and
                                                                  `parent_commit`: who it came from
    <commit>/model_patch.diff                                     what THIS member changed in the
                                                                  harness, relative to its parent
    <commit>/predictions/<commit>_0/{instance}.md                 its full rollout on this instance
    <commit>/predictions/<commit>_0/{instance}.json               the fix it produced for the
                                                                  instance, in "model_patch"
    <commit>/logs/run_evaluation/<commit>_0/<commit>_0/{instance}/report.json   per-test results

Two different things are called `model_patch`. `<commit>/model_patch.diff` is the harness the member
runs; the `model_patch` inside the prediction json is that member's attempt at the instance. The
rollout shows the second and never the first.

# Earlier readings, if there are any

Your working directory may also hold the output of distills that ran before you on this same run:

    distill_<commit>.json                  what that distill wrote, raw
    _distill/<commit>/verdict_<commit>.json  the same after repair, with the counts and `instance`
    _distill/<commit>/trajectory_<commit>.json  what that distill did to reach it

They are readings by an agent like you, of the same evolving population, but almost always on a
DIFFERENT instance -- each generation draws its own. Check `instance` before you use one, or follow
`<commit>/metadata.json`'s `entry` if it is an older file that does not carry the field.

They are worth reading and they are not the answer. What is useful in one is what the population
had already tried and what came of it, which you cannot see from your own instance alone. What is
not useful is its conclusion: it was reached on other rollouts, and a second agent restating it
adds nothing that the step reading you can act on. If you end up naming the same evolution point,
name it because your own instance shows it, and say what your instance adds.

# What the harness is made of

`_agent_src/` is where every member started. It is NOT what any of them is running now: a member's
harness is that tree with its own `model_patch.diff` applied, and its parent's before that, back to
`initial`. If something you want to say turns on what a member's harness actually does, reconstruct
it -- copy `_agent_src`, follow `parent_commit` back to `initial`, then `patch -p1` each diff oldest
first -- rather than reading `_agent_src` and assuming. Reconstruct only when you need to; reading
the rollouts is usually enough to see what happened.

What can be changed is exactly these:

    harness_def.py    instruction text appended to what the agent is told before it solves, and
                      the policy for a reply that comes back unusable. Read the file.
    tools/            what the agent can call.
    coding_agent.py   the episode itself: how the task is handed over, how many attempts it gets,
                      what is checked between them, and what happens when the conversation ends.
                      Read the file.
    utils/

Everything else, including the model, its token limits and the loop in llm_withtools.py, is fixed.

Before you say the harness is missing something, check whether the members you are describing have
already tried it. A gap that is real in `_agent_src` may have been addressed generations ago. But an
attempt is not a fix: a member can add a function that never fires, a threshold whose value makes it
a no-op, or an instruction the model does not follow. If they have tried it, that is worth more than
finding a fresh gap -- say what they tried and what the rollouts show it did, because "they built
this and it still ends the same way" points at a sharper evolution point than "this is missing".

# What to look for

Read across ALL members and count, in the run's own five outcomes: how many solved it, how many
failed the tests, how many ended with an empty patch, how many with a test-only patch, how many
left no trajectory. Then how long the rollouts are and how they ended. Then ask what about the
harness those numbers point at.

An evolution point is one of:
  - an instruction surface that is empty, or says something the population plainly is not doing
  - a tool that is missing, or shaped so that using it costs more than it should
  - what happens to a single reply that comes back unusable: whether it is asked for again, how
    many times, what the model is told when it is, what of the cut-off reply is carried over
  - something the episode does not check before it accepts what the agent produced, or does not do
    when what the agent produced is unusable

The last two are the ones worth separating, because the same counts can point at either and they
fix different amounts.

The reply one acts while the conversation is still going, and only on a reply that was cut off. The
episode one acts after the conversation has stopped, on whatever state the tree is in, however it
stopped and whether or not anything was edited.

So a population whose rollouts stop MID-SENTENCE can be read either way, and a population whose
rollouts stop normally having edited nothing can only be read the second way. If the rollouts you
counted stop mid-sentence, say so and say which reading the counts support -- how many of the
cut-off ones still produced a patch is the number that separates them. Both files are yours to
name; neither is more correct in the abstract.

The episode one is the easier to miss, because the episode is a few lines and says nothing about
itself. Anything you can state as "the episode ended with X and nothing noticed" belongs there.

Two at the right level:

  - "Most of the population produced a patch that changes only test files. Nothing distinguishes
     editing a test from editing the code the test exercises before the patch is accepted."
  - "Rollouts here run three times longer than the population's median and most end without a fix.
     Nothing asks the agent to work in smaller steps or to stop exploring once the cause is known."

Two at the wrong level, both to be avoided:

  - "The fix belongs in django/db/models/sql/query.py" -- that is this issue's content, not the
     harness.
  - "Make the editor's view command return 80 lines instead of the whole file" -- that is the
     change itself. The step that receives your report already has the agent's current source and
     decides what to write.

# What to produce

Write your answer as JSON with exactly these keys:

  "observation":     ONE OR TWO SENTENCES, under 60 words. What the population did and what about
                     the harness that points at.

                     Do NOT restate the tally here. A count computed from the run is printed
                     directly under your sentence, and your own counts go in "evidence". Restating
                     it risks showing the diagnosis two different numbers for the same thing. You
                     are good at saying what you saw and bad at counting it; the counting is
                     already done.

                     Name only the outcomes that actually happened -- listing the ones that did not
                     produces "none left no trajectory" and spends the sentence on nothing. Say
                     what the rollouts ended WITH, in the five outcomes the run itself records:

                         solved              the patch was scored and passed
                         failed the tests    a real patch, scored, wrong
                         an empty patch      the working tree was unchanged
                         a test-only patch   edits, but none to the code under test
                         no trajectory       killed before it wrote anything

                     Those are states of the finished episode, and something can look at a state.
                     Use these words for the outcome itself -- they read as "most rollouts ended
                     with an empty patch", not as column headings to be enumerated.

                     A test-only patch changed no code under test. `remove_patches_to_tests`
                     removes every test edit from a patch before it is graded
                     (swe_bench/report.py), so what reached the scorer was empty and the rollout
                     is recorded in the run's own `total_emptypatch_ids`, beside the ones that
                     changed nothing at all. The counts above keep them apart because how a
                     rollout failed is worth knowing. The scorer does not.

                     Truncation is not one of those five and is a poor substitute for one. A reply
                     cut off at the output limit is common, and a cut-off reply may still have
                     produced a patch. So "they were truncated" does not say whether anything was
                     lost, and a population where every rollout truncated may be one where every
                     rollout also produced a working patch. Lead with the outcome.

                     Truncation does belong here in one case: it is what separates the third kind
                     of evolution point from the fourth. If you are pointing at what happens to an
                     unusable reply rather than at what the episode accepts, the rollouts stopping
                     mid-sentence is your reason and you should say it. Say it as the reason, after
                     the outcome, and put the count in "evidence".
  "candidate_files": 0-3 paths from the list above that your observation points at. Name a file
                     only if it follows from what you measured, and leave it empty rather than
                     reach for the nearest one.

                     But name ALL of them that follow. If the counts support two readings -- the
                     reply policy and what the episode accepts, say -- then both files follow and
                     both belong here. Naming one is a decision, and it is not yours to make: the
                     step that reads you has the full source, has to choose exactly one
                     improvement, and is better served by two options it can weigh than by one it
                     will simply take.

                     This is a guess either way, and it is read as one: you have not run the code.
  "evidence":        one short line of counts you actually measured, in the same words, e.g.
                     "22 members: 3 solved, 0 failed the tests, 13 empty patches,
                     6 test-only patches; 19 of 22 truncated".

Write it to `{outfile}`, not to any other name. Several of these run at once in this same
directory, so a fixed name is a shared slot and the last writer wins. Scratch files go in
`{scratch}`.

You are NOT expected to find something. If the population handled this instance well, say so and
return an empty `candidate_files`. A harness that is not the problem here is the most useful thing
you can report when it is true, and inventing an evolution point on an instance the population
already handles wastes the step that reads you.

When `{outfile}` is written, finish with exactly:

    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

Nothing else ends the episode.
"""

_sync_lock = threading.Lock()
_synced: dict[str, str] = {}

def workspace(run: Path, log=print) -> Path:
    work = ROOT / "probes" / "_distill_ws" / run.name
    key = ",".join(sorted(d.name for d in run.iterdir() if d.is_dir()))
    with _sync_lock:
        if _synced.get(str(work)) == key:
            return work
        work.mkdir(parents=True, exist_ok=True)
        _sync(run, work, log)
        _synced[str(work)] = key
    return work

def _sync(run: Path, work: Path, log) -> None:
    r = subprocess.run(["rsync", "-a", *SYNC_EXCLUDES, f"{run}/", f"{work}/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"rsync failed: {r.stderr[-300:]}")
    for link in (d for d in run.iterdir() if d.is_symlink()):
        if not link.resolve().is_dir():
            log(f"  ! {link.name} is a dangling symlink, skipped")
            continue
        rr = subprocess.run(["rsync", "-aL", *SYNC_EXCLUDES,
                             f"{link.resolve()}/", f"{work/link.name}/"],
                            capture_output=True, text=True)
        log(f"  dereferenced {link.name} -> {link.resolve().name}"
            + ("" if rr.returncode == 0 else f"  (rsync {rr.returncode})"))
    gone = purge_excluded(work)
    log(f"[distill] workspace synced, {len(list(work.glob('*/metadata.json')))} members: {work}"
        + (f" ({gone} eval.sh removed, they embed the gold test patch)" if gone else ""))

ALLOWED_FILES = ("coding_agent.py", "harness_def.py", "tools/", "utils/")
def measured(run: Path, instance: str) -> dict:
    from swe_bench.report import remove_patches_to_tests
    n = res = unres = nodiff = testonly = timeout = 0
    for meta in sorted(run.glob("*/metadata.json")):
        d = meta.parent
        try:
            op = json.loads(meta.read_text()).get("overall_performance") or {}
        except Exception:
            continue
        ids = (set(op.get("total_resolved_ids") or []) | set(op.get("total_unresolved_ids") or [])
               | set(op.get("total_emptypatch_ids") or []))
        if instance not in ids:
            continue
        n += 1
        if instance in set(op.get("total_resolved_ids") or []):
            res += 1; continue
        if instance in set(op.get("total_unresolved_ids") or []):
            unres += 1; continue
        pdirs = list((d / "predictions").iterdir()) if (d / "predictions").is_dir() else []
        raw, traj = "", 0
        if pdirs:
            js, md = pdirs[0] / f"{instance}.json", pdirs[0] / f"{instance}.md"
            traj = md.stat().st_size if md.is_file() else 0
            if js.is_file():
                try:
                    j = json.loads(js.read_text()); j = j[0] if isinstance(j, list) else j
                    raw = j.get("model_patch") or ""
                except Exception:
                    pass
        if traj == 0:            timeout += 1
        elif not raw.strip():    nodiff += 1
        elif not remove_patches_to_tests(raw).strip(): testonly += 1
        else:                    nodiff += 1
    parts = [f"{n} members", f"{res} solved", f"{unres} failed the tests"]
    if nodiff:   parts.append(f"{nodiff} empty patch" + ("es" if nodiff != 1 else ""))
    if testonly: parts.append(f"{testonly} test-only patch" + ("es" if testonly != 1 else ""))
    if timeout:  parts.append(f"{timeout} no trajectory")
    return {"members": n, "solved": res, "rate": round(res / n, 3) if n else None,
            "counts": {"members": n, "solved": res, "failed": unres,
                       "empty": nodiff, "testonly": testonly, "notrajectory": timeout},
            "line": ", ".join(parts)}

def solve_rate(run: Path, instance: str) -> dict:
    members, solved = 0, 0
    for meta in list(run.glob("*/metadata.json")):
        try:
            op = json.loads(meta.read_text()).get("overall_performance") or {}
        except Exception:
            continue
        ids = set(op.get("total_resolved_ids") or []) | set(op.get("total_unresolved_ids") or []) \
              | set(op.get("total_emptypatch_ids") or [])
        if instance not in ids:
            continue
        members += 1
        if instance in set(op.get("total_resolved_ids") or []):
            solved += 1
    return {"members": members, "solved": solved,
            "rate": round(solved / members, 3) if members else None}

def render_block(v: dict, instance: str, neutral: bool = False) -> str:
    sr = v.get("solve_rate") or {}
    rate = "unknown" if sr.get("rate") is None else f"{sr['solved']} of {sr['members']}"
    if neutral:
        head = [
            "# What an agent found in the rollouts on this instance",
            "",
            "Read this first. Below it you will be shown one or two individual rollouts in full.",
            "This block is different: an agent was given the rollouts counted below, read them in",
            "full, and reports one thing it found. It is additional evidence, not a replacement",
            "for the rollouts, and not a decision that has been made for you.",
            "",
            f"Instance: {instance}",
            f"Rollouts it read, and how many solved it: {rate}",
        ]
        measured_label = "Measured over those rollouts"
    else:
        head = [
            "# What the whole population did on this instance",
            "",
            "Read this first. Below it you will be shown one or two individual rollouts in full. This",
            "block is different: an agent read EVERY rollout the population has left on this instance,",
            "counted them, and reports one thing they had in common. It is additional evidence, not a",
            "replacement for the rollouts, and not a decision that has been made for you.",
            "",
            f"Instance: {instance}",
            f"Members of the population that solved it: {rate}",
        ]
        measured_label = "Measured across the population"
    lines = head + [f"Observation: {v.get('observation','')}"]
    if v.get("measured"):
        lines.append(f"{measured_label}: {v['measured']}")
    if v.get("candidate_files"):
        lines.append("Where it guessed this lives, without having run the code: "
                     + ", ".join(v["candidate_files"]))
    lines += [
        "",
        "Judge for yourself whether it holds and whether it matters. The agent's current",
        "implementation is in the system message above. If you conclude it already handles this,",
        "name the function that does and quote the line that makes it fire -- a function existing",
        "is not the same as it running.",
        "",
    ]
    return "\n".join(lines)

def harvest(work: Path, traj: Path, exit_status, outfile: str) -> dict:
    v = {"ok": False, "exit_status": str(exit_status)}
    f = work / outfile
    if not f.is_file():
        v["error"] = f"{outfile} was never written"; return v
    try:
        d = json.loads(f.read_text())
    except Exception as e:
        v["error"] = f"{outfile} is not JSON: {e}"; return v

    obs = str(d.get("observation") or "").strip()
    v["repairs"] = repairs = []

    if "```" in obs:
        obs = re.sub(r"```.*?```", "", obs, flags=re.S).replace("```", "")
        obs = " ".join(obs.split())
        repairs.append("stripped a code fence")
    if "diff --git" in obs:
        obs = " ".join(obs.split("diff --git")[0].split())
        repairs.append("cut at 'diff --git'")
    words = obs.split()
    if len(words) > 80:
        obs = " ".join(words[:80]) + " ..."
        repairs.append(f"clipped from {len(words)} words to 80")

    cand = d.get("candidate_files")
    cand = cand if isinstance(cand, list) else ([cand] if cand else [])
    norm, dropped = [], []
    for c in cand:
        c = str(c).strip().lstrip("./")
        for pre in ("_agent_src/", "initial/"):
            if c.startswith(pre):
                c = c[len(pre):]
        (norm if c.startswith(ALLOWED_FILES) else dropped).append(c)
    if dropped:
        repairs.append("dropped candidate file(s) outside the editable surface: "
                       + ", ".join(dropped[:3]))
    if len(norm) > 3:
        repairs.append(f"kept the first 3 of {len(norm)} candidate files")
        norm = norm[:3]

    v.update(observation=obs, candidate_files=norm, evidence=d.get("evidence"))
    if not obs:
        v["error"] = "the agent wrote no observation"
        return v
    v["ok"] = True
    return v

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("instance")
    ap.add_argument("--model", default=os.getenv("DISTILL_MODEL", "deepseek/deepseek-v4-pro"))
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    run = Path(a.run).resolve()
    work = workspace(run)
    outfile = f"distill_{a.instance}.json"

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.litellm_model import LitellmModel
    out = work / OUT_DIR / a.instance
    out.mkdir(parents=True, exist_ok=True)
    agent = make_tracing_agent(DefaultAgent, False)(
        LitellmModel(model_name=a.model, cost_tracking="ignore_errors",
                     observation_template=OBSERVATION_TEMPLATE),
        LocalEnvironment(cwd=str(work), timeout=60),
        system_template=SYSTEM, instance_template=INSTANCE.format(instance=a.instance),
        step_limit=a.steps, cost_limit=0,
        output_path=out / "trajectory.json")
    result = agent.run(task=a.instance)
    exit_status = result.get("exit_status") if isinstance(result, dict) else result
    (out / "raw_result.txt").write_text(str(result))

    verdict = harvest(work, out / "trajectory.json", exit_status, outfile)
    m = measured(run, a.instance)
    verdict["solve_rate"] = {k: m[k] for k in ("members", "solved", "rate")}
    verdict["measured"] = m["line"]
    verdict["agent_evidence"] = verdict.pop("evidence", None)
    verdict["block"] = render_block(verdict, a.instance)
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    (out / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2))
    print(f"\n  trajectory: {out/'trajectory.json'}")
    return 0 if verdict["ok"] else 1

if __name__ == "__main__":
    sys.exit(main())

def run_for(run_dir, instance: str, tag: str, model: str = "", steps: int = 60,
            log=print) -> str | None:
    try:
        run = Path(run_dir).resolve()
        m0 = measured(run, instance)
        if m0["members"] < MIN_MEMBERS:
            log(f"[distill] {tag} skipped for {instance}: {m0['members']} member(s), "
                f"fewer than {MIN_MEMBERS}; nothing to distill across")
            return None
        work = workspace(run, log)
        outfile = f"distill_{tag}.json"
        scratch = work / SCRATCH_DIR / tag
        scratch.mkdir(parents=True, exist_ok=True)
        out = work / OUT_DIR / tag
        out.mkdir(parents=True, exist_ok=True)

        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment
        from minisweagent.models.litellm_model import LitellmModel
        agent = make_tracing_agent(DefaultAgent, False)(
            LitellmModel(model_name=model or os.getenv("DISTILL_MODEL", "deepseek/deepseek-v4-pro"),
                         cost_tracking="ignore_errors", observation_template=OBSERVATION_TEMPLATE),
            LocalEnvironment(cwd=str(work), timeout=60),
            system_template=SYSTEM,
            instance_template=INSTANCE.format(instance=instance, outfile=outfile,
                                              scratch=str(scratch.relative_to(work))),
            step_limit=steps, cost_limit=0,
            output_path=out / f"trajectory_{tag}.json")

        result = agent.run(task=instance)
        exit_status = result.get("exit_status") if isinstance(result, dict) else result

        v = harvest(work, out / f"trajectory_{tag}.json", exit_status, outfile)
        m = measured(run, instance)
        v["instance"] = instance
        v["solve_rate"] = {k: m[k] for k in ("members", "solved", "rate")}
        v["measured"] = m["line"]
        v["agent_evidence"] = v.pop("evidence", None)
        v["calls"] = getattr(agent, "n_calls", None)
        (out / f"verdict_{tag}.json").write_text(json.dumps(v, ensure_ascii=False, indent=2))

        try:
            keep = run / "_distill" / tag
            keep.mkdir(parents=True, exist_ok=True)
            for src in (out / f"verdict_{tag}.json", out / f"trajectory_{tag}.json",
                        work / outfile):
                if src.is_file():
                    shutil.copy2(src, keep / src.name)
            if scratch.is_dir() and any(scratch.iterdir()):
                shutil.copytree(scratch, keep / "scratch", dirs_exist_ok=True)
        except Exception as e:
            log(f"[distill] could not keep artifacts with the run: {type(e).__name__}: {e}")

        if not v.get("ok"):
            log(f"[distill] {tag} nothing usable for {instance}: "
                f"{v.get('error')}  [{v.get('calls')} calls, {exit_status}]")
            return None
        v["block"] = render_block(v, instance)
        if v.get("repairs"):
            log(f"[distill] {tag} repaired for {instance}: {'; '.join(v['repairs'])}")
        log(f"[distill] {tag} {instance}: {v['solve_rate']['solved']}/{v['solve_rate']['members']}"
            f" solved; {v['observation'][:80]}  [{v.get('calls')} calls, {exit_status}]")
        return v["block"]
    except Exception as e:
        import traceback
        log(f"[distill] failed for {instance}: {type(e).__name__}: {e}")
        log(traceback.format_exc()[-600:])
        return None
