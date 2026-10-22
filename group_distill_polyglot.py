from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

from group_distill import (
    MIN_MEMBERS,
    OBSERVATION_TEMPLATE,
    OUT_DIR,
    SCRATCH_DIR,
    make_tracing_agent,
    render_block,
    workspace,
)

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
                                                                  instance, in "model_patch", plus
                                                                  "success" and "eval_result"
    <commit>/predictions/<commit>_0/{instance}_eval.md            what the exercise's own test
                                                                  command printed against that fix
    <commit>/predictions/<commit>_0/{instance}_docker.log         the container's own log

The exercise's reference solution lives under `.meta/` in the benchmark repository, which is not
here. Nothing in your working directory contains the answer, so `{instance}_eval.md` is a test
report and not a spoiler: read it. It is usually the fastest way to see how a failure failed.

Those directories hold the same three files for EVERY instance the member was scored on, not only
for `{instance}`. Your question is about `{instance}` and that is what you answer about. The one
exception is the `never graded` outcome below: it is a property of the harness rather than of any
exercise, most of it lands on instances no entry can ever be drawn from, and the counts you are
given include the whole run's rate for exactly that reason. If your own instance shows it, the
other instances are there to tell you how widely it holds and which of its shapes dominates.

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

    coding_agent_polyglot.py   the episode itself: how the exercise is handed over, what the
                               agent is told before it starts, how many attempts it gets, what is
                               checked between them, and what happens when the conversation ends.
                               This is the whole instruction surface. There is no separate
                               harness_def on this path -- coding_agent_polyglot.py does not import
                               it -- and coding_agent.py is the SWE-bench episode, which is not
                               what runs here. Read the file.
    tools/                     what the agent can call.
    utils/

One thing about the episode is not in that file and is worth knowing. Before grading, the harness
restores the exercise and keeps only the files the exercise names as its solution; work the agent
left anywhere else is discarded, and the member is then scored on a patch that does not contain it.
A long rollout ending in an empty patch can be that, rather than an agent that did nothing.

Everything else, including the model, its token limits and the loop in llm_withtools.py, is fixed.

Before you say the harness is missing something, check whether the members you are describing have
already tried it. A gap that is real in `_agent_src` may have been addressed generations ago. But an
attempt is not a fix: a member can add a function that never fires, a threshold whose value makes it
a no-op, or an instruction the model does not follow. If they have tried it, that is worth more than
finding a fresh gap -- say what they tried and what the rollouts show it did, because "they built
this and it still ends the same way" points at a sharper evolution point than "this is missing".

# What to look for

Read across ALL members and count, in the run's own five outcomes: how many solved it, how many
failed the tests, how many ended with an empty patch, how many were never graded, how many left no
trajectory. Then how long the rollouts are and how they ended. Then ask what about the harness
those numbers point at.

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

  - "Every member committed its work before finishing, so the restore step found nothing to
     stash and the exercise was never graded. Nothing tells the agent to leave its edits in the
     working tree."
  - "Rollouts here run three times longer than the population's median and most end without a fix.
     Nothing asks the agent to work in smaller steps or to stop exploring once the cause is known."

Two at the wrong level, both to be avoided:

  - "The last verse must not end with a trailing blank line" -- that is this exercise's
     content, not the harness.
  - "Make the editor's view command return 80 lines instead of the whole file" -- that is the
     change itself. The step that receives your report already has the agent's current source and
     decides what to write.

# What to produce

Write your answer as JSON with exactly these keys:

  "observation":     ONE OR TWO SENTENCES, under 60 words. What the population did and what about
                     the harness that points at.

                     ★ Two different things end up in that sentence and they are not worth the
                     same. What the rollouts DID is read off the logs and the patches: you saw
                     it. WHY it happened is an inference you are making without having run the
                     code, and it is the part that has been wrong. A distill on this run said a
                     safeguard was a no-op "because its repo_has_tests basename check counts any
                     .java file as a test". The check is `**/*Test*.java`, the exercise ships
                     `src/test/java/SgfParsingTest.java`, and the flag it set was correct: the
                     observed part held and the reason did not, and a diagnosis acting on that
                     reason would have edited a glob that was never the problem.

                     So separate them. State what you saw first, plainly. If you then say WHY,
                     make it visibly a claim about a mechanism -- and QUOTE the line you are
                     reading it from, from a rollout, a log or the source. If you cannot quote a
                     line, say what you saw and stop; a sentence that ends at the observation is
                     more use than one that carries an explanation nothing supports.

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
                         an empty patch      the agent produced no diff at all
                         never graded        a real patch, and eval.sh never ran on it
                         no trajectory       killed before it wrote anything

                     Those are states of the finished episode, and something can look at a state.
                     Use these words for the outcome itself -- they read as "most rollouts ended
                     with an empty patch", not as column headings to be enumerated.

                     `never graded` is the one to look at hardest, and the one you will not
                     find by reading scores. The run puts it in no id list at all, so it is
                     absent from `total_resolved_ids`, `total_unresolved_ids` AND
                     `total_emptypatch_ids`, while `accuracy_score` still counts it against the
                     member. Before grading the harness runs, in the container:

                         git stash push <the files the exercise names as its solution>
                         git reset --hard <test_commit>
                         git clean -fd
                         git stash pop            <- exit 1 stops the script here

                     It fails three ways, and `<instance>_docker.log` says which:

                       "No local changes to save" then "No stash entries found."
                           there was nothing to stash. Either the agent COMMITTED its work
                           instead of leaving it in the working tree -- `git commit --amend` is
                           the usual shape -- or it solved the exercise in files it created
                           rather than the one the exercise names as its solution.
                       "Entry '<file>' not uptodate. Cannot merge."
                           the pop hit a conflict with something the agent left behind.
                       "exit code 124"
                           not this at all: the solving agent hit its own `timeout 600` and
                           wrote nothing. That one is a different evolution point.

                     Read the rollout and say WHICH. A member that committed and a member that
                     never touched the solution file look identical in every number and point at
                     two different changes: one is about what the agent is told to leave behind,
                     the other about whether it is told which file counts.

                     An empty patch, by contrast, is now only what it says: no diff at all.

                     The count you are given carries the whole run's rate alongside this
                     instance's, because an entry is only ever drawn from the ten exercises of
                     the selection subset and this outcome does not respect that boundary. If
                     the run-wide number is large and your own instance shows none of it, say so
                     plainly and answer about what your instance does show -- a rate measured
                     elsewhere is not evidence about the rollouts in front of you. If your
                     instance does show it, the other instances in these same directories are
                     where you find out whether it is one member's habit or the population's.

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
                     "22 members: 3 solved, 6 failed the tests, 4 empty patches, 9 never
                     graded; 8 of the 9 committed before finishing".

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

ALLOWED_FILES = ("coding_agent_polyglot.py", "tools/", "utils/")

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

def _selection_subset() -> set:
    try:
        return set(json.loads((Path(__file__).resolve().parent /
                               "polyglot" / "subsets" / "small.json").read_text()))
    except Exception as e:
        print(f"[distill] polyglot/subsets/small.json is unreadable ({type(e).__name__}: {e}); "
              f"never-graded instances cannot be identified and will not be counted")
        return set()

def never_graded_run_wide(run: Path) -> tuple[int, int]:
    subset = _selection_subset()
    ng = attempts = 0
    for meta in sorted(run.glob("*/metadata.json")):
        try:
            d = json.loads(meta.read_text())
        except Exception:
            continue
        op = d.get("overall_performance") or {}
        if not op:
            continue
        seen = (set(op.get("total_resolved_ids") or []) | set(op.get("total_unresolved_ids") or [])
                | set(op.get("total_emptypatch_ids") or []))
        attempts += len(subset)
        ng += len(subset - seen)
        deep = d.get("overall_performance_deep") or {}
        if deep:
            dseen = (set(deep.get("total_resolved_ids") or [])
                     | set(deep.get("total_unresolved_ids") or [])
                     | set(deep.get("total_emptypatch_ids") or []))
            n = deep.get("total_submitted_instances") or 0
            attempts += n
            ng += max(0, n - len(dseen))
    return ng, attempts

def measured(run: Path, instance: str) -> dict:
    subset = _selection_subset()
    n = res = unres = nodiff = timeout = incomplete = 0
    for meta in sorted(run.glob("*/metadata.json")):
        d = meta.parent
        try:
            op = json.loads(meta.read_text()).get("overall_performance") or {}
        except Exception:
            continue
        if not op:
            continue
        ids = (set(op.get("total_resolved_ids") or []) | set(op.get("total_unresolved_ids") or [])
               | set(op.get("total_emptypatch_ids") or []))
        if instance not in ids:
            if instance not in subset:
                continue
            n += 1; incomplete += 1; continue
        n += 1
        if instance in set(op.get("total_resolved_ids") or []):
            res += 1; continue
        if instance in set(op.get("total_unresolved_ids") or []):
            unres += 1; continue
        pdirs = list((d / "predictions").iterdir()) if (d / "predictions").is_dir() else []
        traj = 0
        if pdirs:
            md = pdirs[0] / f"{instance}.md"
            traj = md.stat().st_size if md.is_file() else 0
        if traj == 0:
            timeout += 1
        else:
            nodiff += 1
    parts = [f"{n} members", f"{res} solved", f"{unres} failed the tests"]
    if nodiff:     parts.append(f"{nodiff} empty patch" + ("es" if nodiff != 1 else ""))
    if incomplete: parts.append(f"{incomplete} never graded")
    if timeout:    parts.append(f"{timeout} no trajectory")
    line = ", ".join(parts)
    ng, att = never_graded_run_wide(run)
    if att:
        line += (f". Across the whole run, {ng} of {att} attempts on all instances ended never "
                 f"graded" + (" -- none on this one" if ng and not incomplete else ""))
    return {"members": n, "solved": res, "rate": round(res / n, 3) if n else None,
            "counts": {"members": n, "solved": res, "failed": unres, "empty": nodiff,
                       "incomplete": incomplete, "notrajectory": timeout,
                       "never_graded_run_wide": ng, "attempts_run_wide": att},
            "line": line}

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
        v["block"] = render_block(v, instance, neutral=True)
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

def main() -> int:
    ap = argparse.ArgumentParser(description="The Polyglot group distill, over one run.")
    ap.add_argument("run"); ap.add_argument("instance")
    ap.add_argument("--model", default=os.getenv("DISTILL_MODEL", "deepseek/deepseek-v4-pro"))
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    run = Path(a.run).resolve()
    work = workspace(run)
    outfile = f"distill_{a.instance}.json"
    scratch = work / SCRATCH_DIR / a.instance
    scratch.mkdir(parents=True, exist_ok=True)

    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.models.litellm_model import LitellmModel
    out = work / OUT_DIR / a.instance
    out.mkdir(parents=True, exist_ok=True)
    agent = make_tracing_agent(DefaultAgent, False)(
        LitellmModel(model_name=a.model, cost_tracking="ignore_errors",
                     observation_template=OBSERVATION_TEMPLATE),
        LocalEnvironment(cwd=str(work), timeout=60),
        system_template=SYSTEM,
        instance_template=INSTANCE.format(instance=a.instance, outfile=outfile,
                                          scratch=str(scratch.relative_to(work))),
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
    verdict["block"] = render_block(verdict, a.instance, neutral=True)
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    (out / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2))
    print(f"\n  trajectory: {out/'trajectory.json'}")
    return 0 if verdict["ok"] else 1

if __name__ == "__main__":
    sys.exit(main())
