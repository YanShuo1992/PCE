from __future__ import annotations

import hashlib
import json
import platform
import re

from dataclasses import dataclass
from typing import Any, Union, cast

from polyglot.constants import (
    MAP_REPO_TO_INSTALL,
    MAP_REPO_VERSION_TO_SPECS,
    USE_X86,
)

from polyglot.dockerfiles import (
    get_dockerfile_base,
    get_dockerfile_env,
    get_dockerfile_instance,
)
from swebench.harness.utils import (
    get_requirements,
    get_environment_yml,
)

DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"

@dataclass
class TestSpec:
    instance_id: str
    repo: str
    repo_script_list: list[str]
    eval_script_list: list[str]
    env_script_list: list[str]
    arch: str

    @property
    def setup_env_script(self):
        return "\n".join(["#!/bin/bash", "set -euxo pipefail"] + self.env_script_list) + "\n"

    @property
    def eval_script(self):
        return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + self.eval_script_list) + "\n"

    @property
    def install_repo_script(self):
        return "\n".join(["#!/bin/bash", "set -euxo pipefail"] + self.repo_script_list) + "\n"

    @property
    def base_image_key(self):
        return f"pb.base.{self.arch}:latest"

    @property
    def env_image_key(self):
        hash_object = hashlib.sha256()
        hash_object.update(str(self.env_script_list).encode("utf-8"))
        hash_value = hash_object.hexdigest()
        val = hash_value[:22]
        return f"pb.env.{self.arch}.{val}:latest"

    @property
    def instance_image_key(self):
        return f"pb.eval.{self.arch}.{self.instance_id}:latest"

    def get_instance_container_name(self, run_id=None):
        if not run_id:
            return f"pb.eval.{self.instance_id}"
        return f"pb.eval.{self.instance_id}.{run_id}"

    @property
    def base_dockerfile(self):
        return get_dockerfile_base(self.platform, self.arch)

    @property
    def env_dockerfile(self):
        return get_dockerfile_env(self.platform, self.arch)

    @property
    def instance_dockerfile(self):
        return get_dockerfile_instance(self.platform, self.env_image_key)

    @property
    def platform(self):
        if self.arch == "x86_64":
            return "linux/x86_64"
        elif self.arch == "arm64":
            return "linux/arm64/v8"
        else:
            raise ValueError(f"Invalid architecture: {self.arch}")

def get_test_specs_from_dataset(dataset: Union[list[dict], list[TestSpec]]) -> list[TestSpec]:
    if isinstance(dataset[0], TestSpec):
        return cast(list[TestSpec], dataset)
    return list(map(make_test_spec, cast(list[dict], dataset)))

def make_repo_script_list(specs, repo, repo_directory, base_commit, env_name):
    setup_commands = [
        f"cd /{repo}",
        f"git clone . {repo_directory}",
        f"chmod -R 777 {repo_directory}",
        f"cd {repo_directory}",
        f"git reset --hard {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        'echo "Current environment: $CONDA_DEFAULT_ENV"',
    ]
    if repo in MAP_REPO_TO_INSTALL:
        setup_commands.append(MAP_REPO_TO_INSTALL[repo])

    if "pre_install" in specs:
        for pre_install in specs["pre_install"]:
            setup_commands.append(pre_install)

    if "install" in specs:
        setup_commands.append(specs["install"])
    return setup_commands

def replace_uninstallable_packages_requirements_txt(requirement_str: str) -> str:
    replacements = {
        "types-pkg_resources": "types-pkg-resources==0.1.3",
    }
    requirements = [req.strip() for req in requirement_str.split("\n") if req.strip()]
    requirements_replaced = []
    for requirement in requirements:
        if requirement in replacements:
            print(f"Replaced {requirement!r} with {replacements[requirement]!r} (replace_uninstallable_packages)")
            requirements_replaced.append(replacements[requirement])
        else:
            requirements_replaced.append(requirement)
    return "\n".join(requirements_replaced) + "\n"

def make_env_script_list(instance: dict, specs: dict, env_name: str) -> list[str]:
    HEREDOC_DELIMITER = "EOF_59812759871"
    reqs_commands = [
        "source /opt/miniconda3/bin/activate",
    ]
    pkgs = specs.get("packages", "")
    if pkgs == "requirements.txt":
        cmd = f"conda create -n {env_name} python={specs['python']} -y"
        reqs_commands.append(cmd)

        reqs = replace_uninstallable_packages_requirements_txt(get_requirements(instance))
        path_to_reqs = "$HOME/requirements.txt"
        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        cmd = f"conda activate {env_name} && python -m pip install -r {path_to_reqs}"
        reqs_commands.append(cmd)
        reqs_commands.append(f"rm {path_to_reqs}")
    elif pkgs == "environment.yml":
        reqs = get_environment_yml(instance, env_name)
        path_to_reqs = "environment.yml"
        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        if "no_use_env" in specs and specs["no_use_env"]:
            cmd = f"conda create -c conda-forge -n {env_name} python={specs['python']} -y"
            reqs_commands.append(cmd)

            cmd = f"conda env update -f {path_to_reqs}"
            reqs_commands.append(cmd)
        else:
            cmd = f"conda env create --file {path_to_reqs}"
            reqs_commands.append(cmd)

            cmd = f"conda activate {env_name} && conda install python={specs['python']} -y"
            reqs_commands.append(cmd)

        reqs_commands.append(f"rm {path_to_reqs}")
    else:
        cmd = f"conda create -n {env_name} python={specs['python']} {pkgs} -y"
        reqs_commands.append(cmd)

    reqs_commands.append(f"conda activate {env_name}")

    if "pip_packages" in specs:
        pip_packages = " ".join(specs["pip_packages"])
        cmd = f"python -m pip install {pip_packages}"
        reqs_commands.append(cmd)
    return reqs_commands

def make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch):
    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)

    added_files = re.findall(r'^diff --git a/(\S+) b/\1\nnew file mode', test_patch, re.MULTILINE)
    modified_files = [f for f in test_files if f not in added_files]

    commands = []
    if modified_files:
        commands.append(f"git checkout {base_commit} -- {' '.join(modified_files)}")
    if added_files:
        commands.append(f"rm -f {' '.join(added_files)}")
    if instance['language'] == 'cpp':
        commands.append("rm -rf build/")

    reset_tests_command = " && ".join(commands) if commands else "true"

    apply_test_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )
    language = instance['language']
    test_command = MAP_REPO_VERSION_TO_SPECS[language]["test_cmd"]
    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",
        f"cd {repo_directory}",
        "git status",
        "git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        test_command,
        reset_tests_command,
    ]
    return eval_commands

def make_test_spec(instance: dict) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    instance_id = instance['instance_id']
    language = instance['language']
    repo = instance['repo']
    base_commit = instance["base_commit"]
    test_patch = instance["test_patch"]

    env_name = "testbed"
    repo_directory = f"/{env_name}"
    specs = MAP_REPO_VERSION_TO_SPECS[language]

    repo_script_list = make_repo_script_list(specs, repo, repo_directory, base_commit, env_name)
    env_script_list = make_env_script_list(instance, specs, env_name)
    eval_script_list = make_eval_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )
    if platform.machine() in {"aarch64", "arm64"}:
        arch = "arm64" if instance_id not in USE_X86 else "x86_64"
    else:
        arch = "x86_64"

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        arch=arch,
    )
