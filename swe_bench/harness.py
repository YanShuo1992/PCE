import argparse
import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import docker
from datasets import load_dataset

from prompts.testrepo_prompt import get_test_description
from swebench.harness.test_spec import make_test_spec
from swebench.harness.docker_build import build_env_images, build_container, cleanup_container

from swe_bench.utils import (
    copy_to_container,
    copy_from_container,
    log_container_output,
    remove_existing_container,
    safe_log,
    setup_logger,
)
from utils.common_utils import load_json_file

from agent_src import source_root

GEA_ROOT = Path(__file__).resolve().parent.parent

def process_entry(entry, out_dname, model_name_or_path, model_patch_paths):
    instance_id = entry['instance_id']
    problem_statement = entry['problem_statement']
    base_commit = entry['base_commit']
    chat_history_file = out_dname / (instance_id + ".md")
    out_fname = out_dname / (instance_id + ".json")
    eval_file = out_dname / f"{instance_id}_eval.sh"

    if out_fname.exists():
        print(f"Skipping existing entry {instance_id}")
        return {"success": True, "instance_id": instance_id}

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

        logger.info(f"Using GEA project root: {GEA_ROOT}")

        src = source_root()
        copy_to_container(container, src / 'coding_agent.py', '/dgm/coding_agent.py')
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

        logger.info("Setting up environment")
        eval_script = test_spec.eval_script
        eval_file.write_text(eval_script)
        copy_to_container(container, eval_file, '/eval.sh')
        exec_result = container.exec_run("/bin/bash /eval.sh", workdir='/')
        log_container_output(exec_result)
        exec_result = container.exec_run("rm /eval.sh", workdir='/')
        log_container_output(exec_result)

        test_description = get_test_description(eval_script=eval_script, swerepo=True)

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
            "FORBID_ANSWER_LOOKUP": os.getenv('FORBID_ANSWER_LOOKUP'),
            "PATCH_RULES": os.getenv('PATCH_RULES'),
            "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
            "ZAI_API_KEY": os.getenv('ZAI_API_KEY'),
            "AWS_REGION": os.getenv('AWS_REGION'),
            "AWS_REGION_NAME": os.getenv('AWS_REGION_NAME'),
            "AWS_ACCESS_KEY_ID": os.getenv('AWS_ACCESS_KEY_ID'),
            "AWS_SECRET_ACCESS_KEY": os.getenv('AWS_SECRET_ACCESS_KEY'),
            "CODING_AGENT_CLAUDE_MODEL": coding_agent_claude_model,
        }
        if coding_agent_claude_model:
            safe_log(f"Using Claude model for coding agent: {coding_agent_claude_model}")
        else:
            safe_log("Using default Claude model for coding agent: Opus 4.5")
        safe_log("Running the agent")
        cmd = [
            "timeout", os.getenv('SWE_SOLVE_TIMEOUT', '4200'),
            "python", "/dgm/coding_agent.py",
            "--problem_statement", problem_statement,
            "--git_dir", "/testbed/",
            "--chat_history_file", chat_history_file_container,
            "--base_commit", base_commit,
            "--outdir", "/dgm/",
            "--test_description", test_description,
            "--instance_id", instance_id,
        ]
        exec_result = container.exec_run(cmd, environment=env_vars, workdir='/')
        log_container_output(exec_result, raise_error=False)

        agent_failed = False
        failure_reason = None
        if exec_result.exit_code == 124:
            agent_failed = True
            failure_reason = "timeout"
            logger.warning(f"Agent execution timed out for {instance_id}")
        elif exec_result.exit_code and exec_result.exit_code != 0:
            agent_failed = True
            failure_reason = f"exit_code_{exec_result.exit_code}"
            logger.warning(f"Agent execution failed with exit code {exec_result.exit_code} for {instance_id}")

        logger.info("Copying output files back to host")
        try:
            copy_from_container(container, chat_history_file_container, chat_history_file)
            exec_result = container.exec_run(f"find /dgm/ -name '{instance_id}_*.md'", workdir='/')
            chat_history_files_container = exec_result.output.decode().split()
            for chat_history_file_container in chat_history_files_container:
                chat_history_file = out_dname / Path(chat_history_file_container).name
                copy_from_container(container, chat_history_file_container, chat_history_file)
        except Exception as e:
            logger.warning(f"Failed to copy chat history files: {e}")

        model_patch = ""
        patch_recovered = False
        proposed_model_patches = []
        try:
            logger.info("Getting model_patch")
            exec_result = container.exec_run("cat /dgm/model_patch.diff")
            log_container_output(exec_result, raise_error=False)
            if exec_result.exit_code == 0:
                model_patch = exec_result.output.decode()
            elif os.getenv('RECOVER_PATCH_ON_TIMEOUT', '').strip().lower() in ('1', 'true', 'yes', 'on'):
                logger.info("model_patch.diff missing -> recovering the working tree from /testbed")
                recover_cmd = (
                    "python -c \"import sys; sys.path.insert(0, '/dgm'); "
                    "from utils.git_utils import diff_versus_commit; "
                    f"sys.stdout.write(diff_versus_commit('/testbed/', '{base_commit}'))\""
                )
                exec_result = container.exec_run(["/bin/sh", "-c", recover_cmd])
                if exec_result.exit_code == 0:
                    recovered = exec_result.output.decode()
                    if recovered.strip():
                        model_patch = recovered
                        patch_recovered = True
                        logger.info(f"Recovered a {len(recovered)}-char patch for {instance_id}")
                    else:
                        logger.info(f"Nothing to recover for {instance_id}: the working tree is clean")
                else:
                    logger.warning(f"Recovery failed for {instance_id}: {exec_result.output.decode()[:500]}")
            exec_result = container.exec_run("find /dgm/ -name 'model_patch_*.diff'", workdir='/')
            if exec_result.exit_code == 0:
                model_patch_files_container = exec_result.output.decode().split()
                for model_patch_file_container in model_patch_files_container:
                    exec_result = container.exec_run(f"cat {model_patch_file_container}")
                    log_container_output(exec_result, raise_error=False)
                    if exec_result.exit_code == 0:
                        proposed_model_patch = exec_result.output.decode()
                        proposed_model_patches.append(proposed_model_patch)
        except Exception as e:
            logger.warning(f"Failed to get model_patch: {e}")

        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            'proposed_model_patches': proposed_model_patches,
        }

        if patch_recovered:
            result["patch_recovered"] = True

        if agent_failed:
            result["success"] = False
            result["failure_reason"] = failure_reason
            result["exit_code"] = exec_result.exit_code
        else:
            result["success"] = True

        out_fname.write_text(json.dumps(result, indent=4))
        logger.info(f"Generated JSON file for {instance_id} (success: {not agent_failed})")

        return {"success": not agent_failed, "instance_id": instance_id}

    except Exception as e:
        print(f"Error processing entry {instance_id}: {str(e)}")
        if 'logger' in locals():
            logger.error(f"Exception processing entry {instance_id}: {str(e)}")

        try:
            result = {
                "instance_id": instance_id,
                "model_name_or_path": model_name_or_path,
                "model_patch": "",
                "proposed_model_patches": [],
                "success": False,
                "failure_reason": "exception",
                "error": str(e),
            }
            out_fname.write_text(json.dumps(result, indent=4))
            if 'logger' in locals():
                logger.info(f"Generated JSON file for failed entry {instance_id}")
        except Exception as json_error:
            if 'logger' in locals():
                logger.error(f"Failed to generate JSON file for {instance_id}: {json_error}")

        return {"success": False, "instance_id": instance_id, "error": str(e)}

    finally:
        try:
            if 'container' in locals() and 'client' in locals():
                cleanup_logger = logger if 'logger' in locals() else None
                cleanup_container(client, container, cleanup_logger)
        except Exception as e:
            print(f"Error cleaning up Docker container for {instance_id}: {e}")

def harness(
        test_task_list=None,
        num_samples=-1,
        max_workers=4,
        model_name_or_path=None,
        model_patch_paths=None,
        num_evals=1,
        num_evals_parallel=1,
        pred_dname='./swe_bench/predictions',
    ):
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified")
    dataset = dataset['test']

    if model_name_or_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_or_path = f"{timestamp}--claude-3-5-haiku-20241022"
    pred_dname = Path(pred_dname)
    pred_dname.mkdir(exist_ok=True)
    out_dnames = []

    entries = list(dataset)
    if test_task_list:
        entries = [entry for entry in entries if entry['instance_id'] in test_task_list]
    if num_samples > 0:
        entries = entries[:num_samples]

    client = docker.from_env()
    build_env_images(client, dataset=entries, force_rebuild=False, max_workers=max_workers)

    def process_evaluation(eval_idx):
        model_name_or_path_inst = f"{model_name_or_path}_{eval_idx}"
        out_dname = pred_dname / model_name_or_path_inst
        out_dname.mkdir(exist_ok=True)

        print(f"Starting evaluation {eval_idx} for model {model_name_or_path}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_entry = {
                executor.submit(process_entry, entry, out_dname, model_name_or_path_inst, model_patch_paths): entry
                for entry in entries
            }

            for future in as_completed(future_to_entry):
                result = future.result()
                if result["success"]:
                    print(f"Successfully processed entry {result['instance_id']} for eval {eval_idx}")
                else:
                    print(f"Failed to process entry {result['instance_id']} for eval {eval_idx}: {result.get('error', 'Unknown error')}")
        return out_dname

    with ThreadPoolExecutor(max_workers=num_evals_parallel) as eval_executor:
        out_dnames = list(eval_executor.map(process_evaluation, range(num_evals)))

    print(f"All evaluations completed for model {model_name_or_path}")
    return out_dnames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of concurrent threads")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Model name or path")
    parser.add_argument("--model_patch_paths", type=str, default=None, help="Paths to the model patches")
    parser.add_argument("--num_evals", type=int, default=1, help="Repeated number of swe evaluations")
    parser.add_argument("--num_evals_parallel", type=int, default=1, help="Number of parallel repeated evaluations")
    parser.add_argument("--pred_dname", type=str, default="./swe_bench/predictions", help="Output directory for predictions")
    parser.add_argument("--test_task_list", type=str, default=None, help="Subset of swe issues to process")
    args = parser.parse_args()

    if args.test_task_list == 'small':
        test_task_list = load_json_file("./swe_bench/subsets/small.json")
    elif args.test_task_list == 'medium':
        test_task_list = load_json_file("./swe_bench/subsets/medium.json")
    else:
        test_task_list = None

    harness(
        test_task_list=test_task_list,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        model_name_or_path=args.model_name_or_path,
        model_patch_paths=args.model_patch_paths,
        num_evals=args.num_evals,
        num_evals_parallel=args.num_evals_parallel,
        pred_dname=args.pred_dname,
    )

if __name__ == "__main__":
    main()
