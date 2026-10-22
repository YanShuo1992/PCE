from typing import Any

def build_bootstrap_instruction() -> str:
    return ""

def build_execution_instruction() -> str:
    return ""

def build_verification_instruction() -> str:
    return ""

def build_failure_recovery_instruction() -> str:
    return ""

def truncation_retry_budget() -> int:
    return 0

def build_truncation_retry_message() -> str:
    return ""

def keep_partial_text_on_retry() -> bool:
    return True

DEFAULT_TRUNCATION_RETRY_MESSAGE = (
    "Your previous reply was cut off at the output token limit before it contained a "
    "complete tool call. Keep this reply short: skip the explanation and issue the "
    "single next tool call."
)

def compose_instruction_blocks() -> str:
    blocks = [
        ("bootstrap", build_bootstrap_instruction()),
        ("execution", build_execution_instruction()),
        ("verification", build_verification_instruction()),
        ("failure_recovery", build_failure_recovery_instruction()),
    ]
    parts = [f"\n\n<{name}>\n{text.strip()}\n</{name}>"
             for name, text in blocks if text and text.strip()]
    return "".join(parts)

SURFACES = {
    "bootstrap_instruction": build_bootstrap_instruction,
    "execution_instruction": build_execution_instruction,
    "verification_instruction": build_verification_instruction,
    "failure_recovery_instruction": build_failure_recovery_instruction,
}

_EXPLICITLY_ALLOWED_FILES = frozenset()

_ALLOWED = (
    "coding_agent.py",
    "harness_def.py",
    "tools/",
    "tests/",
    "utils/",
)

def allowed_edit_prefixes() -> tuple:
    return _ALLOWED

ALLOWED_EDIT_PREFIXES = _ALLOWED

FORBIDDEN_EDIT_PREFIXES = (
    "prompts/",
    "utils/eval_utils.py",
    "utils/evo_utils.py",
    "utils/docker_utils.py",
    "utils/swe_log_parsers.py",
    "model_io.py",
    "llm_withtools.py",
    "swe_bench/",
    "polyglot/",
    "llm.py",
    "GEA_outer.py",
    "self_improve_step.py",
    "requirements.txt",
)

def patch_target_paths(diff_text: str) -> list[str]:
    paths = set()
    for line in diff_text.splitlines():
        for pre in ("+++ ", "--- "):
            if not line.startswith(pre):
                continue
            p = line[len(pre):].strip().split("\t")[0]
            if p == "/dev/null":
                continue
            if p.startswith(("a/", "b/")):
                p = p[2:]
            paths.add(p)
    return sorted(paths)

def check_patch_paths(diff_text: str) -> tuple[bool, str]:
    paths = patch_target_paths(diff_text)
    if not paths:
        return True, "empty patch"
    bad = []
    for p in paths:
        if p in _EXPLICITLY_ALLOWED_FILES:
            continue
        if p.startswith(FORBIDDEN_EDIT_PREFIXES):
            bad.append(f"{p} (forbidden)")
        elif not p.startswith(allowed_edit_prefixes()):
            bad.append(f"{p} (not in allowlist)")
    if bad:
        return False, "edits outside the evolvable surface: " + ", ".join(bad)
    return True, f"{len(paths)} path(s) within surface"
