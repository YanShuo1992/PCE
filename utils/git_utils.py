import os
import git
import subprocess

def get_git_commit_hash(repo_path='.'):
    try:
        repo = git.Repo(repo_path)
        commit_hash = repo.head.commit.hexsha
        return commit_hash
    except Exception as e:
        print("Error while getting git commit hash:", e)
        return None

def apply_patch(git_dname, patch_str):
    cmd = ["git", "-C", git_dname, "apply", "--reject", "-"]
    result = subprocess.run(
        cmd,
        input=patch_str,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    if result.returncode != 0:
        print(f"apply_patch error: Patch did not fully apply. Return code: {result.returncode}, stdout: {result.stdout}, stderr: {result.stderr}")
    else:
        print("apply_patch successful")

def diff_versus_commit(git_dname, commit):
    diff_cmd = ["git", "-C", git_dname, "diff", commit]
    result = subprocess.run(diff_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    diff_output = result.stdout.decode()

    untracked_files_cmd = ["git", "-C", git_dname, "ls-files", "--others", "--exclude-standard"]
    result = subprocess.run(untracked_files_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    untracked_files = result.stdout.decode().splitlines()

    for file in untracked_files:
        file_path = os.path.join(git_dname, file)
        devnull = '/dev/null'
        if os.name == 'nt':
            devnull = 'NUL'
        diff_file_cmd = ["git", "-C", git_dname, "diff", "--no-index", devnull, file]
        result = subprocess.run(
            diff_file_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=git_dname,
            check=False
        )
        diff_file_output = result.stdout.decode('utf-8', errors='replace')
        diff_output += diff_file_output

    return diff_output

def reset_to_commit(git_dname, commit):
    reset_cmd = ["git", "-C", git_dname, "reset", "--hard", commit]
    result_reset = subprocess.run(
        reset_cmd,
        capture_output=True,
        text=True,
        check=False
    )
    if result_reset.returncode != 0:
        print(f"reset_to_commit error: Failed to reset {git_dname} to commit '{commit}'. STDOUT: {result_reset.stdout} STDERR: {result_reset.stderr}")
    else:
        print(f"reset_to_commit successful: {commit}")

    clean_cmd = ["git", "-C", git_dname, "clean", "-fd"]
    result_clean = subprocess.run(
        clean_cmd,
        capture_output=True,
        text=True,
        check=False
    )
    if result_clean.returncode != 0:
        print(f"reset_to_commit clean error: Failed to clean {git_dname}. STDOUT: {result_clean.stdout} STDERR: {result_clean.stderr}")
    else:
        print(f"reset_to_commit clean successful: {commit}")

def filter_patch_by_files(patch_str, target_files):
    lines = patch_str.splitlines()
    filtered_lines = []
    include_block = False

    for line in lines:
        if line.startswith("diff --git"):
            include_block = any(f"a/{target}" in line and f"b/{target}" in line for target in target_files)
        if include_block:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)

def remove_patch_by_files(patch_str, keyword='polyglot'):
    lines = patch_str.splitlines()
    filtered_lines = []
    include_block = True

    for line in lines:
        if line.startswith("diff --git"):
            include_block = keyword.lower() not in line.lower()
        if include_block:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)
