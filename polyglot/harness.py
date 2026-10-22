import argparse
import datetime
import json
import os
import tempfile
from enum import Enum
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import docker
from datasets import load_dataset

from prompts.testrepo_prompt import get_test_description
from polyglot.test_spec import make_test_spec
from polyglot.docker_build import build_env_images, build_container, cleanup_container
from polyglot.constants import MAP_REPO_VERSION_TO_SPECS, TEST_COMMANDS
from utils.git_utils import filter_patch_by_files, remove_patch_by_files

from swe_bench.utils import (
    copy_to_container,
    copy_from_container,
    log_container_output,
    remove_existing_container,
    safe_log,
    setup_logger,
)

from agent_src import source_root

GEA_ROOT = Path(__file__).resolve().parent.parent

def get_eval_script(commands):
    return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + commands) + "\n"

def process_entry(entry, out_dname, model_name_or_path, model_patch_paths):
    instance_id = entry['instance_id']
    problem_statement = entry['problem_statement']
    base_commit = entry['base_commit']
    chat_history_file = out_dname / (instance_id + ".md")
    out_fname = out_dname / (instance_id + ".json")
    eval_file = out_dname / f"{instance_id}_eval.sh"
    eval_result_file = out_dname / f"{instance_id}_eval.md"

    if out_fname.exists():
        print(f"Skipping existing entry {instance_id}")
        with open(out_fname) as f:
            result = json.loads(f.read())
        return result

    try:
        client = docker.from_env()
        run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        logger = setup_logger(str(out_dname / f"{instance_id}_docker.log"))
        nocache = True
        test_spec = make_test_spec(entry)
        container_name = test_spec.get_instance_container_name(run_id)
        remove_existing_container(client, container_name)
        container = build_container(test_spec, client, run_id, logger, nocache, force_rebuild=False)
        container.start()

        src = source_root()
        copy_to_container(container, src / 'coding_agent_polyglot.py', '/dgm/coding_agent.py')
        copy_to_container(container, src / 'harness_def.py', '/dgm/harness_def.py')
        copy_to_container(container, src / 'model_io.py', '/dgm/model_io.py')
        copy_to_container(container, src / 'requirements.txt', '/dgm/requirements.txt')
        copy_to_container(container, src / 'pytest.ini', '/dgm/pytest.ini')
        copy_to_container(container, src / 'tools/', '/dgm/tools/')
        copy_to_container(container, src / 'utils/', '/dgm/utils/')
        copy_to_container(container, src / 'tests/', '/dgm/tests/')
        copy_to_container(container, src / 'prompts/', '/dgm/prompts/')
        copy_to_container(container, src / 'llm.py', '/dgm/llm.py')
        copy_to_container(container, src / 'llm_withtools.py', '/dgm/llm_withtools.py')
        chat_history_file_container = f'/dgm/{chat_history_file.name}'

        exec_result = container.exec_run("ls -R /testbed", workdir='/')
        log_container_output(exec_result)

        eval_cmd = MAP_REPO_VERSION_TO_SPECS[entry['language']]['test_cmd']
        test_description = get_test_description(eval_cmd, polyglot=True)

        if model_patch_paths:
            safe_log("Applying model patches")
            for model_patch_path in model_patch_paths:
                copy_to_container(container, model_patch_path, '/dgm/parent_patch.txt')
                exec_result = container.exec_run("/bin/sh -c 'patch -p1 < /dgm/parent_patch.txt'", workdir='/dgm')
                log_container_output(exec_result)
                exec_result = container.exec_run("rm /dgm/parent_patch.txt", workdir='/dgm')
                log_container_output(exec_result)

        safe_log("Installing more requirements")
        exec_result = container.exec_run("python -m pip install -r /dgm/requirements.txt", workdir='/')
        log_container_output(exec_result)

        import net_policy
        if not net_policy.seal(container, log=logger.info):
            logger.info("[net_policy] episode NOT sealed: treat its score as network-exposed")

        coding_agent_claude_model = os.getenv('CODING_AGENT_CLAUDE_MODEL')
        env_vars = {
            "ANTHROPIC_API_KEY": os.getenv('ANTHROPIC_API_KEY'),
            "DEEPSEEK_API_KEY": os.getenv('DEEPSEEK_API_KEY'),
            "DEEPSEEK_ANTHROPIC_BASE_URL": os.getenv('DEEPSEEK_ANTHROPIC_BASE_URL'),
            "DEEPSEEK_THINKING": os.getenv('DEEPSEEK_THINKING'),
            "CODING_MAX_TOKENS": os.getenv('CODING_MAX_TOKENS'),
            "TRUNCATION_RETRIES": os.getenv('TRUNCATION_RETRIES'),
            "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
            "ZAI_API_KEY": os.getenv('ZAI_API_KEY'),
            "AWS_REGION": os.getenv('AWS_REGION'),
            "AWS_REGION_NAME": os.getenv('AWS_REGION_NAME'),
            "AWS_ACCESS_KEY_ID": os.getenv('AWS_ACCESS_KEY_ID'),
            "AWS_SECRET_ACCESS_KEY": os.getenv('AWS_SECRET_ACCESS_KEY'),
            "CODING_AGENT_CLAUDE_MODEL": coding_agent_claude_model,
            "FORBID_ANSWER_LOOKUP": os.getenv('FORBID_ANSWER_LOOKUP'),
        }
        if coding_agent_claude_model:
            safe_log(f"Using Claude model for coding agent: {coding_agent_claude_model}")
        else:
            safe_log("Using default Claude model for coding agent: Opus 4.5")
        safe_log("Running the agent")
        cmd = [
            "timeout", os.getenv('POLYGLOT_SOLVE_TIMEOUT', '600'),
            "python", "/dgm/coding_agent.py",
            "--problem_statement", problem_statement,
            "--git_dir", "/testbed/",
            "--chat_history_file", chat_history_file_container,
            "--base_commit", base_commit,
            "--outdir", "/dgm/",
            "--test_description", test_description,
            "--language", entry['language'],
        ]
        exec_result = container.exec_run(cmd, environment=env_vars, workdir='/testbed/')
        log_container_output(exec_result)

        logger.info("Copying output files back to host")
        copy_from_container(container, chat_history_file_container, chat_history_file)
        exec_result = container.exec_run(f"find /dgm/ -name '{instance_id}_*.md'", workdir='/')
        chat_history_files_container = exec_result.output.decode().split()
        for chat_history_file_container in chat_history_files_container:
            chat_history_file = out_dname / Path(chat_history_file_container).name
            copy_from_container(container, chat_history_file_container, chat_history_file)

        logger.info("Getting model_patch")
        exec_result = container.exec_run("cat /dgm/model_patch.diff")
        log_container_output(exec_result)
        model_patch = ''
        model_patch = exec_result.output.decode()
        if model_patch and not model_patch.endswith('\n'):
            model_patch = model_patch + '\n'

        proposed_model_patches = []

        eval_result = ''
        if not model_patch:
            eval_result = 'empty_patch'
            result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            'proposed_model_patches': proposed_model_patches,
            "eval_result": eval_result,
            "success": True
        }
            out_fname.write_text(json.dumps(result, indent=4))
            return {"success": True, "instance_id": instance_id, "eval_result": eval_result}

        exec_result = container.exec_run("git -C /testbed stash push " + " ".join(entry['files']['solution']), workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run(f"git -C /testbed reset --hard {entry['test_commit']}", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run(f"git -C /testbed clean -fd", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("git -C /testbed stash pop", workdir='/')
        log_container_output(exec_result)

        safe_log("Running the eval")
        language = entry['language']
        test_command = TEST_COMMANDS[language]

        eval_file.write_text(get_eval_script(test_command))

        copy_to_container(container, eval_file, '/testbed/eval.sh')
        exec_result = container.exec_run("ls -R /testbed", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("chmod +x /testbed/eval.sh", workdir='/')
        log_container_output(exec_result)

        exec_result = container.exec_run("timeout 120 ./eval.sh", workdir='/testbed')
        log_container_output(exec_result, raise_error=False)
        eval_result_file.write_text(exec_result.output.decode())
        if exec_result.exit_code == 0:
            eval_result = 'resolved'
        else:
            eval_result = 'unresolved'

        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            'proposed_model_patches': proposed_model_patches,
            "eval_result": eval_result,
            "success": True
        }
        out_fname.write_text(json.dumps(result, indent=4))

        return {"success": True, "instance_id": instance_id, "eval_result": eval_result}

    except Exception as e:

        if 'eval_result' not in locals():
            eval_result = 'incomplete'
        else:
            eval_result = 'error'
        if 'model_patch' not in locals():
            model_patch = ''
        if 'proposed_model_patches' not in locals():
            proposed_model_patches = ''

        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            'proposed_model_patches': proposed_model_patches,
            "eval_result": eval_result,
            "success": False
        }
        out_fname.write_text(json.dumps(result, indent=4))

        print(f"Error processing entry {instance_id}: {str(e)}")
        return {"success": False, "instance_id": instance_id, "eval_result": eval_result}

    finally:
        try:
            cleanup_container(client, container, logger)
        except Exception as e:
            print(f"Error cleaning up Docker container for {instance_id}: {e}")

def harness(
        dataset_path="polyglot/polyglot_benchmark_metadata.json",
        test_task_list=None,
        num_samples=-1,
        max_workers=4,
        model_name_or_path=None,
        model_patch_paths=None,
        num_evals=1,
        num_evals_parallel=1,
        pred_dname='./polyglot/predictions',
        output_dir='./polyglot/predictions'
    ):
    if model_patch_paths:
        for model_patch_path in model_patch_paths:
            with open(model_patch_path, 'r') as f:
                patch_content = f.read()
            patch_content = remove_patch_by_files(patch_content)
            if patch_content.strip():
                lines = patch_content.splitlines()
                patch_content = '\n'.join(lines) + '\n'
            with open(model_patch_path, 'w') as f:
                f.write(patch_content)

    if num_evals > 1:
        raise ValueError("Multiple evaluations (num_evals > 1) is not supported with polyglot")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)

    if model_name_or_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_or_path = f"{timestamp}--claude-3-5-haiku-20241022"
    pred_dname = Path(pred_dname)
    pred_dname.mkdir(exist_ok=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_dnames = []

    entries = list(dataset)
    if test_task_list:
        entries = [entry for entry in entries if entry['instance_id'] in test_task_list]
    if num_samples > 0:
        entries = entries[:num_samples]

    client = docker.from_env()
    build_env_images(client, dataset=entries, max_workers=max_workers, force_rebuild=False)

    def process_evaluation(eval_idx):
        model_name_or_path_inst = f"{model_name_or_path}_{eval_idx}"
        out_dname = pred_dname / model_name_or_path_inst
        out_dname.mkdir(exist_ok=True)

        print(f"Starting evaluation {eval_idx} for model {model_name_or_path}")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_entry = {
                executor.submit(process_entry, entry, out_dname, model_name_or_path_inst, model_patch_paths): entry
                for entry in entries
            }

            for future in as_completed(future_to_entry):
                result = future.result()
                results.append(future.result())
                if result["success"]:
                    print(f"Successfully processed entry {result['instance_id']} for eval {eval_idx}")
                else:
                    print(f"Failed to process entry {result['instance_id']} for eval {eval_idx}: {result.get('error', 'Unknown error')}")

        return out_dname, results

    out_dname, results = process_evaluation(0)
    print(f"All evaluations completed for model {model_name_or_path}")

    incomplete_ids = [result["instance_id"] for result in results if not result["success"]]
    completed_ids = [result["instance_id"] for result in results if result["success"]]
    resolved_ids = []
    unresolved_ids = []
    error_ids = []
    empty_patch_ids = []
    unstopped_containers = []
    unremoved_images = []

    for result in results:
        if result["success"]:
            if result.get("eval_result") == "resolved":
                resolved_ids.append(result["instance_id"])
            elif result.get("eval_result") == "unresolved":
                unresolved_ids.append(result["instance_id"])
            elif result.get("eval_result") == "empty_patch":
                empty_patch_ids.append(result["instance_id"])
            else:
                error_ids.append(result["instance_id"])

    report = {
        "total_instances": len(dataset),
        "submitted_instances": len(results),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "unstopped_instances": len(unstopped_containers),
        "completed_ids": list(sorted(completed_ids)),
        "incomplete_ids": list(sorted(incomplete_ids)),
        "empty_patch_ids": list(sorted(empty_patch_ids)),
        "submitted_ids": list(sorted(result["instance_id"] for result in results)),
        "resolved_ids": list(sorted(resolved_ids)),
        "unresolved_ids": list(sorted(unresolved_ids)),
        "error_ids": list(sorted(error_ids)),
        "unstopped_containers": list(sorted(unstopped_containers)),
        "unremoved_images": list(sorted(unremoved_images)),
        "schema_version": 2,
    }

    print(report)
    report_file = output_dir / Path(
        model_name_or_path.replace("/", "__") + f"_{0}"
        + f".{000}"
        + ".json"
    )
    with open(report_file, "w") as f:
        print(json.dumps(report, indent=4), file=f)
    print(f"Report written to {report_file}")

    return out_dnames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of concurrent threads")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Model name or path")
    parser.add_argument("--model_patch_paths", type=str, default=None, help="Paths to the model patches")
    parser.add_argument("--num_evals", type=int, default=1, help="Repeated number of swe evaluations")
    parser.add_argument("--num_evals_parallel", type=int, default=1, help="Number of parallel repeated evaluations")
    args = parser.parse_args()

    with open("polyglot/polyglot_benchmark_metadata.json") as f:
        metadata = json.loads(f.read())
        language_task_list = [entry["instance_id"] for entry in metadata if entry["instance_id"].startswith("python")]
        all_task_list = [entry["instance_id"] for entry in metadata]

    from utils.common_utils import load_json_file
    swe_issues_med = load_json_file("./polyglot/subsets/medium.json")
    model_patch_paths = args.model_patch_paths.split(',') if args.model_patch_paths is not None else None

    harness(
        dataset_path="polyglot/polyglot_benchmark_metadata.json",
        test_task_list=all_task_list,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        model_name_or_path=args.model_name_or_path,
        model_patch_paths=model_patch_paths,
        num_evals=args.num_evals,
        num_evals_parallel=args.num_evals_parallel,
    )

if __name__ == "__main__":
    main()
