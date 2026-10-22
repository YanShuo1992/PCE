import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

def load_predictions(paths):
    prediction_paths = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            prediction_paths.append(path)
        elif path.is_dir():
            prediction_paths += list(path.glob("*.json"))
        else:
            assert False, path

    predictions = dict()
    for fname in prediction_paths:
        try:
            pred = json.loads(fname.read_text())
        except json.decoder.JSONDecodeError as err:
            raise err

        if "instance_id" not in pred:
            print("Skipping json without instance_id", fname)
            continue

        inst = pred["instance_id"]
        pred["json_fname"] = str(fname)
        predictions[inst] = pred

    return predictions

def remove_patches_to_tests(model_patch):
    lines = model_patch.splitlines(keepends=True)
    filtered_lines = []
    is_tests = False

    for line in lines:
        if line.startswith("diff --git a/"):
            pieces = line.split()
            to = pieces[-1]
            if to.startswith("b/") and (
                    "/test/" in to
                    or "/tests/" in to
                    or "/testing/" in to
                    or "/test_" in to
                    or "/tox.ini" in to
            ):
                is_tests = True
            else:
                is_tests = False

        if not is_tests:
            filtered_lines.append(line)

    return "".join(filtered_lines)

def preds_to_jsonl(dname, predictions):
    dname = Path(dname)

    predictions_jsonl = str(dname / "all_preds.jsonl")
    model_name_or_path = list(predictions.values())[0]["model_name_or_path"]
    with open(predictions_jsonl, "w") as fh:
        for inst, pred in predictions.items():
            assert model_name_or_path == pred["model_name_or_path"]
            minimal_pred = dict(
                model_name_or_path=model_name_or_path,
                model_patch=remove_patches_to_tests(pred["model_patch"]),
                instance_id=pred["instance_id"],
            )
            fh.write(json.dumps(minimal_pred) + "\n")
    return predictions_jsonl

def run_evals(predictions_jsonl, run_id, dataset_name, root_dir, output_dir, num_eval_procs=5):
    run_evals_cmd = f"""
python {os.path.join(root_dir, 'swe_bench/SWE-bench/swebench/harness/run_evaluation.py')}
    --dataset_name {dataset_name}
    --predictions_path {predictions_jsonl}
    --max_workers {num_eval_procs}
    --run_id {run_id}
"""
    run_evals_cmd = " ".join([line.strip() for line in run_evals_cmd.split() if line.strip()])
    log = Path(output_dir) / f"run_evals.{run_id}.log"
    lines = []
    proc = subprocess.Popen(run_evals_cmd.split(), cwd=output_dir, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1)
    try:
        with proc.stdout:
            for line in proc.stdout:
                lines.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    try:
        log.write_text(f"$ {run_evals_cmd}\n\n--- output ---\n{''.join(lines)}")
    except Exception:
        pass
    if rc != 0:
        tail = "".join(lines[-25:])
        raise subprocess.CalledProcessError(
            rc, run_evals_cmd,
            output=f"see {log}\n{tail}")

def repair_report(output_dir, run_id, predictions_jsonl):
    out = Path(output_dir)
    preds = [json.loads(l) for l in Path(predictions_jsonl).read_text().splitlines() if l.strip()]
    predictions = {p["instance_id"]: p for p in preds}
    if not predictions:
        return None
    model = list(predictions.values())[0]["model_name_or_path"].replace("/", "__")
    rdir = out / "logs" / "run_evaluation" / run_id / model
    if not rdir.is_dir():
        return None

    resolved, unresolved, completed, empty, error = set(), set(), set(), set(), set()
    for iid, pred in predictions.items():
        if pred.get("model_patch", None) in ["", None]:
            empty.add(iid)
            continue
        f = rdir / iid / "report.json"
        if not f.is_file():
            error.add(iid)
            continue
        try:
            rep = json.loads(f.read_text())
        except Exception:
            error.add(iid)
            continue
        completed.add(iid)
        (resolved if rep.get(iid, {}).get("resolved") else unresolved).add(iid)

    if not completed:
        return None

    report = {
        "total_instances": 500,
        "submitted_instances": len(predictions),
        "completed_instances": len(completed),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(unresolved),
        "empty_patch_instances": len(empty),
        "error_instances": len(error),
        "unstopped_instances": 0,
        "completed_ids": sorted(completed),
        "incomplete_ids": [],
        "empty_patch_ids": sorted(empty),
        "submitted_ids": sorted(predictions.keys()),
        "resolved_ids": sorted(resolved),
        "unresolved_ids": sorted(unresolved),
        "error_ids": sorted(error),
        "unstopped_containers": [],
        "unremoved_images": [],
        "schema_version": 2,
        "report_repaired": True,
    }
    (out / f"{model}.{run_id}.json").write_text(json.dumps(report, indent=4))
    return report

def make_report(
        dnames,
        run_ids=None,
        dataset_name="princeton-nlp/SWE-bench_Verified",
        output_dir='./swe_bench',
        dnames_workers=None,
        num_eval_procs=5,
    ):
    root_dir = os.path.abspath(os.getcwd())
    output_dir = os.path.join(root_dir, output_dir)

    def process_single_dname(dname, run_id):
        dname = Path(os.path.join(root_dir, dname))
        predictions = load_predictions([dname])
        predictions_jsonl = preds_to_jsonl(dname, predictions)
        model = list(predictions.values())[0]["model_name_or_path"].replace("/", "__")
        summary = Path(output_dir) / f"{model}.{run_id}.json"
        before = summary.stat().st_mtime if summary.is_file() else None

        err = None
        try:
            run_evals(predictions_jsonl, run_id, dataset_name, root_dir, output_dir, num_eval_procs=num_eval_procs)
        except Exception as e:
            err = e

        fresh = summary.is_file() and (before is None or summary.stat().st_mtime > before)
        if not fresh:
            rep = repair_report(output_dir, run_id, predictions_jsonl)
            if rep is None:
                raise RuntimeError(
                    f"{summary.name} was not written and there is nothing to rebuild it from "
                    f"({dname})") from err
            print(f"  ! {summary.name} was not written; rebuilt from "
                  f"{rep['completed_instances']} per-instance reports "
                  f"({rep['resolved_instances']}/{rep['submitted_instances']} resolved)"
                  + (f"; run_evals said: {err}" if err else ""))
        print(f"Report generated for {dname}")

    if run_ids is None or len(run_ids) != len(dnames):
        run_ids = [f"{i:03}" for i in range(len(dnames))]
    if dnames_workers is None:
        dnames_workers = len(dnames)
    with ThreadPoolExecutor(max_workers=dnames_workers) as executor:
        futures = [executor.submit(process_single_dname, d, r)
                   for d, r in zip(dnames, run_ids)]
        for f in futures:
            f.result()

    print("All reports generated.")

def main():
    parser = argparse.ArgumentParser(description="Run evaluations on predictions.")
    parser.add_argument('--dnames', type=str, nargs='+', help="List of directories of predictions to evaluate.")
    parser.add_argument('--run_ids', type=str, nargs='+', default=None, help="Run ID for this evaluation run.")
    parser.add_argument('--dataset_name', type=str, default="princeton-nlp/SWE-bench_Verified", help="Name of the dataset to evaluate on.")
    parser.add_argument('--dnames_workers', type=int, default=None, help="Number of parallel workers to use for processing dnames.")
    parser.add_argument('--num_eval_procs', type=int, default=5, help="Number of parallel processes to use for evaluation.")
    parser.add_argument('--output_dir', type=str, default='./swe_bench', help="Output directory for the reports.")
    args = parser.parse_args()

    make_report(
        args.dnames,
        run_ids=args.run_ids,
        dataset_name=args.dataset_name,
        dnames_workers=args.dnames_workers,
        num_eval_procs=args.num_eval_procs,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
