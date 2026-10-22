import argparse
import re
import datetime
import json
import os
from pathlib import Path
import docker

from llm import create_client, get_response_from_llm, extract_json_between_markers
from prompts.self_improvement_prompt import get_diagnose_prompt_polyglot, get_diagnose_prompt_swe, get_problem_description_prompt
from prompts.testrepo_prompt import get_test_description
from swe_bench.harness import harness
from polyglot.harness import harness as polyglot_harness
from swe_bench.report import make_report
from utils.common_utils import load_json_file, ensure_patch_file_newline
from utils.evo_utils import get_model_patch_paths, get_all_performance, is_compiled_self_improve, build_task_success_vector
from utils.docker_utils import (
    build_dgm_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    remove_existing_container,
    setup_logger,
    safe_log,
)

dataset = None
diagnose_model = 'o1-2024-12-17'

def get_bedrock_model_name(claude_model):
    model_mapping = {
        'claude_haiku_4.5': 'bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0',
        'claude_sonnet_4.5': 'bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    }
    return model_mapping.get(claude_model, claude_model)

def _get_task_logs_for_parent(dgm_output_dir, parent_commit, task_id):
    try:
        from prompts.self_improvement_prompt import find_selfimprove_eval_logs, process_selfimprove_eval_logs

        if parent_commit == 'initial':
            return f"""
Task ID: {task_id}
Parent Commit: {parent_commit}

Note: This is the initial agent, so no specific task logs are available.
The initial agent represents the baseline DGM codebase before any improvements.
"""

        parent_output_dir = os.path.join(dgm_output_dir, parent_commit)

        md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(
            task_id, parent_output_dir, commit_id=parent_commit
        )

        md_log, eval_log, predicted_patch, eval_result = process_selfimprove_eval_logs(
            md_logs, eval_logs, predicted_patches, eval_results
        )

        combined_log = f"""
Task ID: {task_id}
Parent Commit: {parent_commit}

Agent Execution Log:
{md_log}

Test Results:
{eval_log}

Predicted Patch:
{predicted_patch}

Evaluation Result:
{eval_result}
"""
        return combined_log

    except Exception as e:
        return f"Task ID: {task_id}\nParent Commit: {parent_commit}\nError retrieving task logs: {str(e)}"

def _distill_block(out_dir_base, entry, run_id, polyglot=False, *, root_dir=None,
                   patch_files=None, entry1=None, commit=None, commit1=None,
                   patch_files1=None, dataset=None):
    if not (os.getenv('DISTILL') or '').strip():
        return None
    if not entry or str(entry).startswith('solve_'):
        safe_log(f"[distill] skipped: entry is {entry!r}, not an instance")
        return None
    try:
        if polyglot:
            import group_distill_polyglot as _distill
        else:
            import group_distill as _distill
        return _distill.run_for(out_dir_base, entry, tag=run_id,
                                model=os.getenv('DISTILL_MODEL', ''),
                                steps=int(os.getenv('DISTILL_STEPS', '60') or 60),
                                log=safe_log)
    except Exception as e:
        safe_log(f"[distill] unavailable: {type(e).__name__}: {e}")
        return None

def diagnose_problem(entry, commit, root_dir, out_dir, patch_files=[], max_attempts=3, polyglot=False, group_improve=False, entry1=None, commit1=None, patch_files1=None, coding_agent=None, diagnose_model=None, distill_block=None):
    diagnose_model_param = diagnose_model
    if diagnose_model_param:
        model_to_use = get_bedrock_model_name(diagnose_model_param)
    else:
        model_to_use = globals()['diagnose_model']
    safe_log(f"Using model for problem diagnosis: {model_to_use}")
    client = create_client(model_to_use)

    if group_improve:
        if polyglot:
            diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_polyglot(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
            group_improve=True,
            entry1_id=entry1,
            commit1=commit1,
            patch_files1=patch_files1,
            distill_block=distill_block,
        )
        else:
            diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_swe(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
            group_improve=True,
            entry1_id=entry1,
            commit1=commit1,
            patch_files1=patch_files1,
            distill_block=distill_block,
        )
    elif polyglot:
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_polyglot(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
            distill_block=distill_block,
        )
    else:
        safe_log(f"entry: {entry}")
        safe_log(f"commit: {commit}")
        safe_log(f"root_dir: {root_dir}")
        safe_log(f"out_dir: {out_dir}")
        safe_log(f"patch_files: {patch_files}")
        safe_log(f"max_attempts: {max_attempts}")
        safe_log(f"polyglot: {polyglot}")
        safe_log(f"dataset: {dataset}")
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_swe(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
            distill_block=distill_block,
        )
    try:
        response, msg_history = get_response_from_llm(
            msg=diagnose_prompt,
            client=client[0],
            model=client[1],
            system_message=diagnose_sys_message,
            print_debug=False,
            msg_history=None,
        )
        safe_log(f"Message history: {msg_history}")
        response_json = extract_json_between_markers(response)
        assert response_json, "empty response json"
        problem_statement = get_problem_description_prompt(response_json, polyglot)
    except Exception as e:
        safe_log(f"Error while diagnosing the problem: {e}")
        if max_attempts > 0:
            return diagnose_problem(
                entry, commit, root_dir, out_dir,
                patch_files=patch_files,
                coding_agent=coding_agent,
                diagnose_model=diagnose_model,
                max_attempts=max_attempts-1,
                polyglot=polyglot,
                distill_block=distill_block,
            )
        else:
            return None
    return problem_statement

def save_metadata(metadata, output_dir):
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

def run_harness_swe(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more):
    safe_log('Start harness')
    test_task_list = [entry] if test_task_list is None else test_task_list
    dnames = harness(
        test_task_list=test_task_list,
        num_samples=-1,
        max_workers=min(int(os.getenv('EVAL_MAX_WORKERS', '10')), len(test_task_list)),
        model_name_or_path=model_name_or_path,
        model_patch_paths=patch_files,
        num_evals=num_evals,
        num_evals_parallel=5,
        pred_dname=os.path.join(output_dir, "predictions"),
    )
    metadata['swe_dnames'] = [str(dn) for dn in dnames]
    safe_log('Start make_report')
    make_report(
        dnames,
        run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
        dataset_name="princeton-nlp/SWE-bench_Verified",
        output_dir=output_dir,
        dnames_workers=5,
    )
    safe_log('Start get_performance')
    performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
    metadata['overall_performance'] = overall_performance
    safe_log("End of evaluation")

    if (overall_performance and \
        test_more_threshold is not None and test_task_list_more is not None and \
            overall_performance.get('total_resolved_instances', 0) >= len(test_task_list) * test_more_threshold):
        safe_log("Start additional evaluation cycle")
        dnames = harness(
            test_task_list=test_task_list_more,
            num_samples=-1,
            max_workers=min(int(os.getenv('EVAL_MAX_WORKERS', '10')), len(test_task_list_more)),
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=5,
            pred_dname=os.path.join(output_dir, "predictions"),
        )
        safe_log('Start make_report more')
        make_report(
            dnames,
            run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
            dataset_name="princeton-nlp/SWE-bench_Verified",
            output_dir=output_dir,
            dnames_workers=5,
        )
        safe_log('Start get_performance')
        performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
        metadata['overall_performance'] = overall_performance
        safe_log("End of evaluation more")

def run_harness_polyglot(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more):
    safe_log('Start harness')
    test_task_list = [entry] if test_task_list is None else test_task_list
    safe_log(f'workers {min(int(os.getenv("EVAL_MAX_WORKERS", "10")), len(test_task_list))}')
    dnames = polyglot_harness(
        test_task_list=test_task_list,
        num_samples=-1,
        max_workers=min(int(os.getenv('EVAL_MAX_WORKERS', '10')), len(test_task_list)),
        model_name_or_path=model_name_or_path,
        model_patch_paths=patch_files,
        num_evals=num_evals,
        num_evals_parallel=min(5, num_evals),
        pred_dname=os.path.join(output_dir, "predictions"),
        output_dir=output_dir
    )
    metadata['swe_dnames'] = [str(dn) for dn in dnames]
    safe_log('Start get_performance')
    performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
    metadata['overall_performance'] = overall_performance
    safe_log("End of evaluation")

    if (overall_performance and \
        test_more_threshold is not None and test_task_list_more is not None and \
            overall_performance.get('total_resolved_instances', 0) >= len(test_task_list) * test_more_threshold):
        safe_log("Start additional evaluation cycle")
        dnames = polyglot_harness(
            test_task_list=test_task_list_more,
            num_samples=-1,
            max_workers=min(int(os.getenv('EVAL_MAX_WORKERS', '10')), len(test_task_list_more)),
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=min(5, num_evals),
            pred_dname=os.path.join(output_dir, "predictions"),
            output_dir=output_dir
        )
        safe_log('Start get_performance')
        performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
        metadata['overall_performance_deep'] = overall_performance
        safe_log("End of evaluation more")

def self_improve(
    parent_commit='initial',
    output_dir='output_selfimprove/',
    force_rebuild=False,
    num_evals=1,
    entry=None,
    test_task_list=None,
    test_more_threshold=None,
    test_task_list_more=None,
    full_eval_threshold=None,
    run_baseline=None,
    polyglot=False,
    group_improve=False,
    coding_agent=None,
    diagnose_model=None
):

    global dataset
    if polyglot:
        with open("polyglot/polyglot_benchmark_metadata.json") as f:
            dataset = json.loads(f.read())
    else:
        from datasets import load_dataset
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
        dataset = dataset['test']

    metadata = {}
    root_dir = os.path.abspath('./')
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir_base = output_dir
    output_dir = os.path.join(root_dir, f"{output_dir}/{run_id}/")
    os.makedirs(output_dir, exist_ok=True)
    metadata['run_id'] = run_id
    metadata['parent_commit'] = parent_commit
    test_task_list_big = load_json_file("./swe_bench/subsets/big.json")

    logger = setup_logger(os.path.join(output_dir, "self_improve.log"))

    image_name = "dgm"
    container_name = f"dgm-container-{run_id}"
    client = docker.from_env()
    remove_existing_container(client, container_name)
    container = build_dgm_container(
        client, root_dir, image_name, container_name,
        force_rebuild=force_rebuild,
    )
    container.start()

    if polyglot:
        exec_result = container.exec_run("rm /dgm/coding_agent.py", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("mv /dgm/coding_agent_polyglot.py /dgm/coding_agent.py", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("rm /dgm/utils/eval_utils.py", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("rm /dgm/utils/swe_log_parsers.py", workdir='/')
        log_container_output(exec_result)
    else:
        exec_result = container.exec_run("rm /dgm/coding_agent_polyglot.py", workdir='/')

    if "+" in parent_commit and "+" in entry:
        group_improve = True
        entry_list = entry.split("+")
        parent_commit_list = parent_commit.split("+")
        parent_commit0 = parent_commit_list[0]
        parent_commit1 = parent_commit_list[1]
        entry0 = entry_list[0]
        entry1 = entry_list[1]
        actual_parent_commit = parent_commit0

        patch_files = get_model_patch_paths(root_dir, os.path.join(output_dir, '../'), actual_parent_commit)
        patch_files1 = get_model_patch_paths(root_dir, os.path.join(output_dir, '../'), parent_commit1)
        if run_baseline not in ['no_selfimprove']:
            for patch_file in patch_files:
                ensure_patch_file_newline(patch_file)
                copy_to_container(container, patch_file, '/dgm/parent_patch.txt')
                exec_result = container.exec_run("/bin/sh -c 'patch -p1 < /dgm/parent_patch.txt'", workdir='/dgm')
                log_container_output(exec_result)
                exec_result = container.exec_run("rm /dgm/parent_patch.txt", workdir='/dgm')
                log_container_output(exec_result)

    else:
        patch_files = get_model_patch_paths(root_dir, os.path.join(output_dir, '../'), parent_commit)
        if run_baseline not in ['no_selfimprove']:
            for patch_file in patch_files:
                ensure_patch_file_newline(patch_file)
                copy_to_container(container, patch_file, '/dgm/parent_patch.txt')
                exec_result = container.exec_run("/bin/sh -c 'patch -p1 < /dgm/parent_patch.txt'", workdir='/dgm')
                log_container_output(exec_result)
                exec_result = container.exec_run("rm /dgm/parent_patch.txt", workdir='/dgm')
                log_container_output(exec_result)

    exec_result = container.exec_run("git init -q", workdir='/dgm/')
    log_container_output(exec_result)
    exec_result = container.exec_run("git add --all", workdir='/dgm/')
    log_container_output(exec_result)
    exec_result = container.exec_run("git -c user.name='user' -c user.email='you@example.com' commit -m 'a nonsense commit message'", workdir='/dgm/')
    log_container_output(exec_result)
    commit_output = exec_result.output.decode('utf-8')
    _m = re.search(r'\[[^\]]*?([0-9a-f]{7,40})\]', commit_output)
    if not _m:
        safe_log(f"Could not read the base commit hash from: {commit_output!r}")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata
    commit_hash = _m.group(1)
    safe_log(f"base commit {commit_hash}")

    exec_result = container.exec_run("python -m pip install -r /dgm/requirements.txt", workdir='/')
    log_container_output(exec_result)

    if entry:
        safe_log(f"Task to improve: {entry}")
        if group_improve:
            _block = _distill_block(out_dir_base, entry0, run_id, polyglot=polyglot,
                                    root_dir=root_dir, patch_files=patch_files,
                                    commit=parent_commit0, entry1=entry1,
                                    commit1=parent_commit1, patch_files1=patch_files1,
                                    dataset=dataset)
            problem_statement = diagnose_problem(entry0, parent_commit0, root_dir, out_dir_base, patch_files=patch_files, polyglot=polyglot, group_improve=True, entry1=entry1, commit1=parent_commit1,patch_files1=patch_files1, coding_agent=coding_agent, diagnose_model=diagnose_model, distill_block=_block)
        else:
            _block = _distill_block(out_dir_base, entry, run_id, polyglot=polyglot,
                                    root_dir=root_dir, patch_files=patch_files,
                                    commit=parent_commit, dataset=dataset)
            problem_statement = diagnose_problem(entry, parent_commit, root_dir, out_dir_base, patch_files=patch_files, polyglot=polyglot, coding_agent=coding_agent, diagnose_model=diagnose_model, distill_block=_block)
        safe_log(f"problem_statement: {problem_statement}")
    else:
        safe_log("No entry provided. Exiting.")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    metadata['entry'] = entry
    metadata['problem_statement'] = problem_statement
    if not problem_statement:
        safe_log("Failed to diagnose the problem statement. Exiting.")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    safe_log("Running self-improvement")
    safe_log(f"[selfimprove] output cap {os.getenv('SELFIMPROVE_MAX_TOKENS') or '20480'}, "
             f"truncation budget {os.getenv('SELFIMPROVE_TRUNCATION_RETRIES') or '3'} "
             f"(the solver keeps CODING_MAX_TOKENS={os.getenv('CODING_MAX_TOKENS')}, "
             f"TRUNCATION_RETRIES={os.getenv('TRUNCATION_RETRIES')})")
    chat_history_file_container = "/dgm/self_evo.md"
    test_description = get_test_description(swerepo=False)
    coding_agent_model = coding_agent if coding_agent else os.getenv('CODING_AGENT_CLAUDE_MODEL')
    env_vars = {
        "ANTHROPIC_API_KEY": os.getenv('ANTHROPIC_API_KEY'),
        "DEEPSEEK_API_KEY": os.getenv('DEEPSEEK_API_KEY'),
        "DEEPSEEK_THINKING": os.getenv('DEEPSEEK_THINKING'),
        "CODING_MAX_TOKENS": os.getenv('SELFIMPROVE_MAX_TOKENS') or '20480',
        "SELFIMPROVE_TRUNCATION_RETRIES": os.getenv('SELFIMPROVE_TRUNCATION_RETRIES') or '3',
        "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
        "ZAI_API_KEY": os.getenv('ZAI_API_KEY'),
        "AWS_REGION": os.getenv('AWS_REGION'),
        "AWS_REGION_NAME": os.getenv('AWS_REGION_NAME'),
        "AWS_ACCESS_KEY_ID": os.getenv('AWS_ACCESS_KEY_ID'),
        "AWS_SECRET_ACCESS_KEY": os.getenv('AWS_SECRET_ACCESS_KEY'),
        "CODING_AGENT_CLAUDE_MODEL": coding_agent_model,
    }
    if coding_agent_model:
        safe_log(f"Using coding agent (self-modification): {coding_agent_model}")
    else:
        safe_log("Using default coding agent (self-modification): Opus 4.5")
    import shlex
    cmd = [
        "timeout", "2700",
        "python", "-u", "/dgm/coding_agent.py",
        "--problem_statement", problem_statement,
        "--git_dir", "/dgm/",
        "--chat_history_file", chat_history_file_container,
        "--base_commit", commit_hash,
        "--outdir", "/dgm/",
        "--test_description", test_description,
        "--self_improve",
    ]
    agent_log_container = "/dgm/agent_run.log"
    wrapped_cmd = ["bash", "-c", " ".join(shlex.quote(str(a)) for a in cmd) + f" > {agent_log_container} 2>&1"]
    exec_result = container.exec_run(wrapped_cmd, environment=env_vars, workdir='/')

    chat_history_file = os.path.join(output_dir, "self_evo.md")
    model_patch_file = os.path.join(output_dir, "model_patch.diff")
    for _src, _dst in [
        (agent_log_container, os.path.join(output_dir, "agent_run.log")),
        (chat_history_file_container, chat_history_file),
        ("/dgm/model_patch.diff", model_patch_file),
    ]:
        try:
            copy_from_container(container, _src, _dst)
        except Exception as _e:
            safe_log(f"(could not copy {_src}: {_e})")

    exit_code = getattr(exec_result, "exit_code", None)
    if exit_code not in (0, None):
        safe_log(f"Agent script exited with code {exit_code}; skipping this self-improve attempt. "
                 f"See {os.path.join(output_dir, 'agent_run.log')} for the traceback.")
        metadata['agent_exit_code'] = exit_code
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    try:
        if not os.path.exists(model_patch_file):
            raise Exception("Model patch file is empty or does not exist")
        with open(model_patch_file, 'r') as f:
            patch_content = f.read()
            if not patch_content.strip():
                raise Exception("Model patch file is empty")
    except Exception as e:
        safe_log(f"Failed to read model patch file: {str(e)}")
        cleanup_container(container)
        save_metadata(metadata, output_dir)
        return metadata

    patch_files.append(model_patch_file)

    cleanup_container(container)

    model_patch_exists = os.path.exists(model_patch_file)
    metadata['model_patch_exists'] = model_patch_exists
    model_patch_notempty = os.path.getsize(model_patch_file) > 0
    metadata['model_patch_notempty'] = model_patch_notempty
    model_name_or_path = run_id
    if model_patch_exists and model_patch_notempty:
        try:
            if not polyglot:
                run_harness_swe(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more)
            else:
                run_harness_polyglot(entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id, test_more_threshold, test_task_list, test_task_list_more)
        except Exception as e:
            safe_log(f"Error while evaluating the self-improvement: {e}")

    if metadata.get('overall_performance'):
        task_list_path = os.path.join(root_dir, 'polyglot' if polyglot else 'swe_bench', 'subsets', 'task.json')
        task_success_vector = build_task_success_vector(metadata['overall_performance'], task_list_path)
        if task_success_vector is not None:
            metadata['task_success_vector'] = task_success_vector

    save_metadata(metadata, output_dir)
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Self-improvement step for the repository.")
    parser.add_argument('--parent_commit', default="initial", type=str, help='Current commit to find the eval results, "initial" if starting from original dgm, else the run_id')
    parser.add_argument('--output_dir', default="./output_selfimprove", type=str, help='Directory to store the output')
    parser.add_argument('--force_rebuild', default=False, action='store_true', help='Force rebuild of the Docker image')
    parser.add_argument('--num_evals', default=1, type=int, help='Repeated number of swe evaluations after self-improvement')
    parser.add_argument('--entry', default="django__django-10999", type=str, help='Task entry to improve')
    parser.add_argument('--test_task_list', default=None, type=str, help='List of tasks to evaluate the self-improvement')
    args = parser.parse_args()

    os.system(f"cp -r initial/ {args.output_dir}")

    metadata = self_improve(
        parent_commit=args.parent_commit,
        output_dir=args.output_dir,
        force_rebuild=args.force_rebuild,
        num_evals=args.num_evals,
        entry=args.entry,
        test_task_list=args.test_task_list,
    )

if __name__ == "__main__":
    main()
