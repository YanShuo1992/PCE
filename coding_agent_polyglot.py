import argparse
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import os
import threading

from llm_withtools import GLM_MODEL, OPENAI_MODEL, chat_with_agent, CLAUDE_MODEL
from utils.git_utils import diff_versus_commit, reset_to_commit, apply_patch

NPM_TEST_COMMANDS = [
    ["sh", "-c", "set -e"],
    ["sh", "-c", "[ ! -e node_modules ] && ln -s /npm-install/node_modules ."],
    ["sh", "-c", "[ ! -e package-lock.json ] && ln -s /npm-install/package-lock.json ."],
    ["sed", "-i", "s/\\bxtest(/test(/g", "*.spec.js"],
    ["npm", "run", "test"]
]

CPP_TEST_COMMANDS = [
    ["sh", "-c", "set -e"],
    ["sh", "-c", "[ ! -d \"build\" ] && mkdir build"],
    ["sh", "-c", "cd build"],
    ["cmake", "-DEXERCISM_RUN_ALL_TESTS=1", "-G", "Unix Makefiles", ".."],
    ["make"],
    ["sh", "-c", "cd ../"]
]

TEST_COMMANDS = {
    "python": [["pytest", "-rA", "--tb=long"]],
    "rust": [["cargo", "test", "--", "--include-ignored"]],
    "go": [["go", "test", "./..."]],
    "javascript": NPM_TEST_COMMANDS,
    "cpp": CPP_TEST_COMMANDS,
    "java": [["./gradlew", "test"]],
}

thread_local = threading.local()

def get_thread_logger():
    return getattr(thread_local, 'logger', None)

def set_thread_logger(logger):
    thread_local.logger = logger

def setup_logger(log_file='./chat_history.md', level=logging.INFO):
    logger = logging.getLogger(f'AgenticSystem-{threading.get_ident()}')
    logger.setLevel(level)

    logger.handlers = []

    file_formatter = logging.Formatter('%(message)s')

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)

    set_thread_logger(logger)

    return logger

def safe_log(message, level=logging.INFO):
    logger = get_thread_logger()
    if logger:
        logger.log(level, message)
    else:
        print(f"Warning: No logger found for thread {threading.get_ident()}")

ANSWER_LOOKUP_RULE = (
    "Solve the exercise using only the code, tests and documentation in {git_tempdir}. "
    "Do not look up how this exercise is solved elsewhere: do not fetch or read a published "
    "reference solution, example solution, or anyone else's answer to it, from the network or "
    "from anywhere outside {git_tempdir}. Consulting one is not solving the exercise. "
    "Installing the dependencies the exercise already needs is fine, and so is reading the "
    "language's own documentation."
)

def answer_lookup_rule(git_tempdir):
    if os.getenv('FORBID_ANSWER_LOOKUP', '').strip().lower() not in ('1', 'true', 'yes', 'on'):
        return ''
    return '\n' + ANSWER_LOOKUP_RULE.format(git_tempdir=git_tempdir) + '\n'

class AgenticSystem:
    def __init__(
            self,
            problem_statement,
            git_tempdir,
            base_commit,
            chat_history_file='./chat_history.md',
            test_description=None,
            self_improve=False,
            language='python'
        ):
        self.problem_statement = problem_statement
        self.git_tempdir = git_tempdir
        self.base_commit = base_commit
        self.chat_history_file = chat_history_file
        self.test_description = test_description
        self.self_improve = self_improve
        self.language = language

        coding_agent_model = os.getenv('CODING_AGENT_CLAUDE_MODEL')
        if coding_agent_model:
            from model_io import get_claude_model
            self.code_model = get_claude_model()
        elif self_improve:
            self.code_model = OPENAI_MODEL
        else:
            self.code_model = GLM_MODEL

        self.logger = setup_logger(chat_history_file)

        with open(chat_history_file, 'w') as f:
            f.write('')

    def get_current_edits(self):
        diff = str(diff_versus_commit(self.git_tempdir, self.base_commit))
        new_msg_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"# Current Repo Edits\n{diff}",
                    }
                ],
            }
        ]
        return new_msg_history

    def forward(self):
        task = f"""I have uploaded a code repository in the directory {self.git_tempdir}. Help solve the following problem.

<problem_description>
{self.problem_statement}
</problem_description>

Your task is to make changes to the files in the {self.git_tempdir} directory to address the <problem_description>. I have already taken care of the required dependencies.
{answer_lookup_rule(self.git_tempdir)}"""
        instruction = f"{task}\n\nPlease analyze the problem description carefully. Then make edits to the code files to complete the instruction."
        init_edit = chat_with_agent(instruction, model=self.code_model, msg_history=[], logging=safe_log)

def main():
    parser = argparse.ArgumentParser(description='Process repository with an agentic system.')
    parser.add_argument('--problem_statement', required=True, help='The problem statement to process')
    parser.add_argument('--git_dir', required=True, help='Path to git repository directory')
    parser.add_argument('--base_commit', required=True, help='Base commit hash to compare against')
    parser.add_argument('--chat_history_file', required=True, help='Path to chat history file')
    parser.add_argument('--outdir', required=False, default="/dgm/", help='Output directory')
    parser.add_argument('--test_description', default=None, required=False, help='Description of how to test the repository')
    parser.add_argument('--self_improve', default=False, action='store_true', help='Whether to self-improve the repository or solving swe')
    parser.add_argument('--language', required=False, default="python", choices=['cpp', 'java', 'python', 'go', 'rust', 'javascript'], help='Task\'s programming language')
    args = parser.parse_args()

    agentic_system = AgenticSystem(
        problem_statement=args.problem_statement,
        git_tempdir=args.git_dir,
        base_commit=args.base_commit,
        chat_history_file=args.chat_history_file,
        test_description=args.test_description,
        self_improve=args.self_improve,
        language=args.language,
    )

    agentic_system.forward()

    model_patch = diff_versus_commit(args.git_dir, args.base_commit)
    if model_patch and not model_patch.endswith('\n'):
        model_patch = model_patch + '\n'
    model_patch_outfile = os.path.join(args.outdir, 'model_patch.diff') if args.outdir else 'model_patch.diff'
    with open(model_patch_outfile, 'w') as f:
        f.write(model_patch)

if __name__ == "__main__":
    main()
