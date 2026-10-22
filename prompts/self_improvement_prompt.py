import os
import json

from utils.common_utils import load_json_file, read_file
from utils.docker_utils import safe_log

coding_agent_summary = """# Coding Agent Summary

- **Main File**: `coding_agent.py`
  - Primary Class: `AgenticSystem`
  - The `forward()` function is the central entry point.
  - Prompts are located either within the `forward()` function or in the `prompts/` directory.
- **Tools**: `tools/`
  - The `tools/` directory contains various tools that LLMs can use to perform specific tasks.
  - Each tool must have a `tool_info()` function that returns a JSON object containing 'name', 'description', and 'input_schema'. The 'input_schema' should be a JSON object containing 'type', 'properties', and 'required'.
  - Each tool must have a `tool_function()` function that takes the arguments defined in input_schema, performs the tool's task, and returns a string.
  - See other tools for reference.
- **Utilities**: `utils/`
  - The `utils/` directory contains utility functions used across the codebase.
{surfaces_section}- **Additional Details**:
  - The agent is very good at automatically utilizing the right available tools at the right time. So do not have an agentic flow that explicitly forces a tool's usage.
  - Common tools, such as file editing and bash commands, are easy for the agent to recognize and use appropriately. However, more complex and niche tools may require explicit instructions in the prompt.
  - Tools should be designed to be as general as possible, ensuring they work across any GitHub repository. Avoid hardcoding repository-specific details or behaviors (e.g., paths).
  - Do not use 'while True' loops in the agent's code. This can cause the agent to get stuck and not respond.
  - Verify the implementation details of helper functions prior to usage to ensure proper integration and expected behavior.
  - Do not install additional packages or dependencies directly. Update `requirements.txt` if new dependencies are required and install them using `pip install -r requirements.txt`.
\n\n"""

coding_agent_summary_polyglot = """# Coding Agent Summary

- **Main File**: `coding_agent.py`
  - Primary Class: `AgenticSystem`
  - The `forward()` function is the central entry point.
  - Prompts are located either within the `forward()` function or in the `prompts/` directory.
- **Tools**: `tools/`
  - The `tools/` directory contains various tools that LLMs can use to perform specific tasks.
  - Each tool must have a `tool_info()` function that returns a JSON object containing 'name', 'description', and 'input_schema'. The 'input_schema' should be a JSON object containing 'type', 'properties', and 'required'.
  - Each tool must have a `tool_function()` function that takes the arguments defined in input_schema, performs the tool's task, and returns a string.
  - See other tools for reference.
- **Utilities**: `utils/`
  - The `utils/` directory contains utility functions used across the codebase.

- **Additional Details**:
  - The coding agent trying to solve a programming task. A task is in one programming language, but the coding agent needs to deal with different languages including C++, Go, Java, JavaScript, Python, and Rust.
  - The agent is very good at automatically utilizing the right available tools at the right time. So do not have an agentic flow that explicitly forces a tool's usage.
  - Be detailed in the prompt about what steps (e.g. implementing tests, refining solutions, etc.) you would like the agent to execute.
  - Common tools, such as file editing and bash commands, are easy for the agent to recognize and use appropriately. However, more complex and niche tools may require explicit instructions in the prompt.
  - Tools should be designed to be as general as possible, ensuring they work across any task. Avoid hardcoding task-specific details or behaviors (e.g., paths or solutions).
  - DO NOT use 'while True' loops in the agent's code IN ANY CASE!! This can cause the agent to get stuck and not respond.
  - Verify the implementation details of helper functions prior to usage to ensure proper integration and expected behavior.
  - **DO NOT create parsing errors tools or functions, collecting raw error messages and letting the agent analyze them will be more efficient.**
\n\n
""" + """ 
### DOC: tool function schema

Carefully consider whether to add/enhance the current tool or edit the workflow in forward()

Pay special attention to making sure that "required" and "type" are always at the correct level of nesting. For example, "required" should be at the same level as "properties", not inside it.
Make sure that every property, no matter how short, has a type and description correctly nested inside it.
Other arguments than you have seen are not permitted. For example, in "edit_line_ranges" with "type": "array", arguments like "minItems" and "maxItems" are not permitted.
\n\n
"""

diagnose_system_message = """Here is the implementation of the coding agent.

# Coding Agent Implementation
----- Coding Agent Implementation Start -----
{code}
----- Coding Agent Implementation End -----

Your task is to identify ONE detailed plan that would improve the agent's coding ability. The improvement should not be specific to any particular GitHub issue or repository.
"""

swe_issue_prompt = "Here is the log for the coding agent trying to solve the GitHub issues but failed."
polyglot_issue_prompt = "Here is the log for the coding agent trying to solve a programming task. A task is in one programming language, but the coding agent needs to deal with different languages including C++, Go, Java, JavaScript, Python, and Rust."

diagnose_prompt = """
# Agent Running Log
----- Agent Running Log Start -----
{md_log}
----- Agent Running Log End -----

# GitHub Issue
The GitHub issue that the agent is trying to solve.
----- GitHub Issue Start -----
{github_issue}
----- GitHub Issue End -----

# Predicted Patch
The agent's predicted patch to solve the issue.
----- Predicted Patch Start -----
{predicted_patch}
----- Predicted Patch End -----

# Private Test Patch
SWE-bench's official private tests to detect whether the issue is solved. This is not available to the agent during evaluation. The agent should try to implement its own tests.
----- Private Test Patch Start -----
{test_patch}
----- Private Test Patch End -----

# Issue Test Results
The test results from SWE-bench using the above official private tests.
----- Issue Test Results Start -----
{eval_log}
----- Issue Test Results End -----

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "log_summarization": Analyze the above logs and summarize how the agent tried to solve the GitHub issue. Note which tools and how they are used, the agent's problem-solving approach, and any issues encountered.
- "potential_improvements": Identify potential improvements to the coding agent that could enhance its coding capabilities. Focus on the agent's general coding abilities (e.g., better or new tools usable across any repository) rather than issue-specific fixes (e.g., tools only usable in one framework). All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature or tool could be added or improved to best implement the proposed improvement. If the proposed feature can be implemented by modifying the existing tools, describe the modifications needed, instead of suggesting a new tool. If the improvement is about how the agent should proceed -- what to read first, how to sequence edits, what would justify believing a change is correct, how to back out after a failure -- then say what to write into the corresponding `harness_def.py` surface, which changes the agent's behaviour on every task without adding any code to run.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

diagnose_prompt_groupimprove = """
# Agent Running Log
----- Agent Running Log Start -----
{md_log}
----- Agent Running Log End -----

# GitHub Issue
The GitHub issue that the agent is trying to solve.
----- GitHub Issue Start -----
{github_issue}
----- GitHub Issue End -----

# Predicted Patch
The agent's predicted patch to solve the issue.
----- Predicted Patch Start -----
{predicted_patch}
----- Predicted Patch End -----

# Private Test Patch
SWE-bench's official private tests to detect whether the issue is solved. This is not available to the agent during evaluation. The agent should try to implement its own tests.
----- Private Test Patch Start -----
{test_patch}
----- Private Test Patch End -----

# Issue Test Results
The test results from SWE-bench using the above official private tests.
----- Issue Test Results Start -----
{eval_log}
----- Issue Test Results End -----
"""

diagnose_prompt_4_last_sentence = """
Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "log_summarization": Analyze the above logs and summarize how the agent tried to solve the GitHub issue. Note which tools and how they are used, the agent's problem-solving approach, and any issues encountered.
- "potential_improvements": Identify potential improvements to the coding agent that could enhance its coding capabilities. Focus on the agent's general coding abilities (e.g., better or new tools usable across any repository) rather than issue-specific fixes (e.g., tools only usable in one framework). All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature or tool could be added or improved to best implement the proposed improvement. If the proposed feature can be implemented by modifying the existing tools, describe the modifications needed, instead of suggesting a new tool. If the improvement is about how the agent should proceed -- what to read first, how to sequence edits, what would justify believing a change is correct, how to back out after a failure -- then say what to write into the corresponding `harness_def.py` surface, which changes the agent's behaviour on every task without adding any code to run.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

diagnose_prompt_emptypatches = """There are some empty patches when attempting to solve GitHub issues. Since the coding agent is stochastic, it may not always produce a patch. Handle cases where the coding agent fails to generate a patch or generates one that only modifies the test cases without editing the primary source code. For example, the simplest solution is to ask the agent to try again.

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to the coding agent's system. All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature could be added or improved to best implement the proposed improvement. If the improvement is about how the agent should proceed -- what to read first, how to sequence edits, what would justify believing a change is correct, how to back out after a failure -- then say what to write into the corresponding `harness_def.py` surface, which changes the agent's behaviour on every task without adding any code to run.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

diagnose_prompt_emptypatches_groupimprove = """There are some empty patches when attempting to solve GitHub issues. Since the coding agent is stochastic, it may not always produce a patch. Handle cases where the coding agent fails to generate a patch or generates one that only modifies the test cases without editing the primary source code. For example, the simplest solution is to ask the agent to try again.

"""

diagnose_prompt_stochasticity = """Since the coding agent is stochastic, it may not produce the correct patch for the given problem statement on the first try. Take into account the agent's stochastic nature and provide a solution to handle such cases. For example, one solution could be to ask the agent to try multiple times and select the best patch. The file `utils/eval_utils.py` contains helper functions to evaluate the generated patches. Giving previous attempts as context to the agent may also help.

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to the coding agent's system. All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature could be added or improved to best implement the proposed improvement. If the improvement is about how the agent should proceed -- what to read first, how to sequence edits, what would justify believing a change is correct, how to back out after a failure -- then say what to write into the corresponding `harness_def.py` surface, which changes the agent's behaviour on every task without adding any code to run.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

diagnose_prompt_stochasticity_groupimprove = """Since the coding agent is stochastic, it may not produce the correct patch for the given problem statement on the first try. Take into account the agent's stochastic nature and provide a solution to handle such cases. For example, one solution could be to ask the agent to try multiple times and select the best patch. The file `utils/eval_utils.py` contains helper functions to evaluate the generated patches. Giving previous attempts as context to the agent may also help.
"""

diagnose_prompt_contextlength = """While the coding agent is attempting to solve GitHub issues, it encounters an error due to the input being too long for the requested model. This error is likely due to the context length exceeding the model's maximum input size. Handle cases where the input is too long for the model. The coding agent is mainly using the file `llm_withtools.py`. LLMs typically have a context window of 200k tokens. Handle context length only if the context window limit is reached and caught as an exception; otherwise, it is okay to leave it as is.

<error_message>
Error in get_response_withtools: Error code: 400 - {'message': 'Input is too long for requested model.'}
</error_message>

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to the coding agent's system. All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature could be added or improved to best implement the proposed improvement. If the improvement is about how the agent should proceed -- what to read first, how to sequence edits, what would justify believing a change is correct, how to back out after a failure -- then say what to write into the corresponding `harness_def.py` surface, which changes the agent's behaviour on every task without adding any code to run.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature and details so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

diagnose_prompt_contextlength_groupimprove = """While the coding agent is attempting to solve GitHub issues, it encounters an error due to the input being too long for the requested model. This error is likely due to the context length exceeding the model's maximum input size. Handle cases where the input is too long for the model. The coding agent is mainly using the file `llm_withtools.py`. LLMs typically have a context window of 200k tokens. Handle context length only if the context window limit is reached and caught as an exception; otherwise, it is okay to leave it as is.

<error_message>
Error in get_response_withtools: Error code: 400 - {'message': 'Input is too long for requested model.'}
</error_message>
"""

diagnose_prompt_1_2_3_last_sentence = """Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to the coding agent's system. All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature could be added or improved to best implement the proposed improvement. If the improvement is about how the agent should proceed -- what to read first, how to sequence edits, what would justify believing a change is correct, how to back out after a failure -- then say what to write into the corresponding `harness_def.py` surface, which changes the agent's behaviour on every task without adding any code to run.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature and details so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

problem_description_prompt = """# To Implement\n\n{implementation_suggestion}\n\n{problem_description}"""

def get_problem_description_prompt(response_json, is_polyglot=False):
    if "analysis" in response_json and "improvement_opportunities" in response_json and "recommended_changes" in response_json:
        implementation_suggestion = response_json["recommended_changes"]
        problem_description = response_json["analysis"]
    else:
        implementation_suggestion = response_json.get("implementation_suggestion", "")
        problem_description = response_json.get("problem_description", "")

    if is_polyglot:
        return coding_agent_summary_polyglot + problem_description_prompt.format(implementation_suggestion=implementation_suggestion, problem_description=problem_description)
    else:
        return agent_summary() + problem_description_prompt.format(implementation_suggestion=implementation_suggestion, problem_description=problem_description)

def read_mdlog_file(filepath, filter=True):
    if not filter:
        return read_file(filepath)

    filter_content = [
        'Error in get_response_withtools',
    ]
    filtered_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if not any(line.startswith(fc) for fc in filter_content):
                filtered_lines.append(line.rstrip('\n'))
    return "\n".join(filtered_lines).strip()

def find_selfimprove_eval_logs(entry, out_dir, commit_id='initial', filter=True):
    predictions_dir = os.path.join(out_dir, commit_id, 'predictions')
    all_preds_folders = [f for f in os.listdir(predictions_dir) if os.path.isdir(os.path.join(predictions_dir, f))]
    prediction_log_files = [os.path.join(predictions_dir, f, f"{entry}.md") for f in all_preds_folders]
    prediction_json_files = [os.path.join(predictions_dir, f, f"{entry}.json") for f in all_preds_folders]
    prediction_log_files = [f for f in prediction_log_files if os.path.exists(f)]
    prediction_json_files = [f for f in prediction_json_files if os.path.exists(f)]
    try_eval_logs = [os.path.join(predictions_dir, f, f"{entry}_eval.md") for f in all_preds_folders]
    try_eval_logs = [f for f in try_eval_logs if os.path.exists(f)]

    md_logs = []
    for file in prediction_log_files:
        md_logs.append(read_mdlog_file(file, filter=filter))
    predicted_patches = []
    eval_results = []
    for json_file in prediction_json_files:
        prediction_data = load_json_file(json_file)
        predicted_patch = prediction_data.get("model_patch", "")
        predicted_patches.append(predicted_patch)
        eval_result = prediction_data.get("eval_result", "")
        eval_results.append(eval_result)
    if not try_eval_logs:
        eval_log_files = [
            os.path.join(out_dir, commit_id, f'logs/run_evaluation/', f, f, entry, 'report.json')
            for f in all_preds_folders
        ]
        eval_log_files = [f for f in eval_log_files if os.path.exists(f)]
        eval_logs = []
        for file in eval_log_files:
            eval_json = load_json_file(file)
            eval_logs.append(get_eval_log_text(eval_json))
    else:
        eval_logs = []
        for file in try_eval_logs:
            print(file)
            eval_logs.append(read_file(file))
    return md_logs, eval_logs, predicted_patches, eval_results

def process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results):
    md_log = md_logs[0] if md_logs else "No logs available."

    if isinstance(md_log, str) and not md_log.strip():
        md_log = ("<no trajectory>\n"
                  "Nothing was recorded for this attempt. Every instance seen in this state has "
                  "been a wall-clock timeout, which is also the one ending that guarantees no "
                  "patch was written. There is no evidence here about how the agent works.")
    eval_log = eval_logs[0] if eval_logs else "No test results available. Assume all tests failed."
    predicted_patch = predicted_patches[0] if predicted_patches else "No predicted patch available. Assume the agent failed."

    _raw = predicted_patches[0] if predicted_patches else ""
    if isinstance(_raw, str) and not _raw.strip():
        predicted_patch = (
            "<empty>\n"
            "The agent produced no patch at all: the trajectory above ends with no edit to the "
            "repository. Nothing was submitted and nothing was tested. This is a missing fix, not "
            "a wrong one."
        )
        if not eval_logs:
            eval_log = ("No test was run. The agent submitted no source change, so there was "
                        "nothing to evaluate. Do not read this as a fix that failed its tests.")
    elif isinstance(_raw, str):
        try:
            import sys as _sys, os as _os
            _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from swe_bench.report import remove_patches_to_tests as _strip
            _kept = _strip(_raw).strip()
        except Exception:
            _kept = _raw.strip()
        if not _kept:
            predicted_patch = (
                _raw.rstrip() + "\n\n"
                "<discarded>\n"
                "The patch above was discarded before scoring. Every file it touches is a test "
                "file or tox.ini, and changes to those are stripped so that a submission cannot "
                "alter the tests it is judged by. After stripping, nothing was left to apply, so "
                "no test was run. The agent edited the tests rather than the source."
            )
            if not eval_logs:
                eval_log = ("No test was run. Everything the agent changed was stripped as a test "
                            "file, so no source change was submitted. Do not read this as a fix "
                            "that failed its tests.")

    if len(md_log) > 250000:
        _keep = 125000
        _elided = len(md_log) - 2 * _keep
        md_log = (
            md_log[:_keep]
            + f"\n<output_head/>\n<elided_chars>{_elided} characters elided from the middle of this trajectory</elided_chars>\n<output_tail/>\n"
            + md_log[-_keep:]
        )

    eval_result = eval_results[0] if eval_results else "No evaluation result available. Assume the agent failed."
    return md_log, eval_log, predicted_patch, eval_result

diagnose_prompt_emptypatches_polyglot = """There are some empty patches when attempting to solve GitHub issues. Since the coding agent is stochastic, it may not always produce a patch. Handle cases where the coding agent fails to generate a patch or generates one that only modifies the test cases without editing the primary source code. For example, the simplest solution is to change the prompt to specifically make sure it called the edit tool.

Please analyze the log below to identify why no code edits were made.

# Agent Running Log
----- Agent Running Log Start -----
{md_log}
----- Agent Running Log End -----

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to the coding agent's system. All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature could be added or improved to best implement the proposed improvement.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

diagnose_prompt_emptypatches_polyglot_groupimprove = """Here is the log of another coding agent — not the current one — that attempted to solve the GitHub issues but failed. You can use it as a reference: There are some empty patches when attempting to solve GitHub issues. Since the other coding agent is stochastic, it may sometimes fail to produce a patch. In such cases, handle situations where the coding agent either fails to generate a patch or produces one that only modifies the test files without editing the main source code. For example, the simplest solution is to ask the agent to try again.

Please analyze the log below to identify why no code edits were made.

# Agent Running Log
----- Agent Running Log Start -----
{md_log}
----- Agent Running Log End -----

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to the coding agent's system. All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature could be added or improved to best implement the proposed improvement.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

diagnose_prompt_stochasticity_polyglot = """Since the coding agent is stochastic, it may not produce the correct patch for the given problem statement on the first try. 
Take into account the agent's stochastic nature and provide a solution to handle such cases. 
For example, one solution could be to ask the agent to try multiple times and select the best patch according to the test results. Scale the reflection times is also a good idea.
Giving previous attempts and test results as context to the agent may also help.
The tests for tasks are not provided in the repo, and the agent needs workflow design to implement them.
Agent's own tests that validate the solution may not cover all the cases that will be checked by the private tests during official scoring. So the quality of implemented tests are crucial.

Please analyze the log below to identify how we can improve the testing process and multiple solution attempts.

# Agent Running Log
----- Agent Running Log Start -----
{md_log}
----- Agent Running Log End -----

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to the coding agent's system. All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature could be added or improved to best implement the proposed improvement.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

diagnose_prompt_stochasticity_polyglot_groupimprove = """Here is the log of another coding agent — not the current one — that attempted to solve the GitHub issues but failed. You can use it as a reference: 
Since the other coding agent is stochastic, it may not produce the correct patch for the given problem statement on the first try. 
Take into account the agent's stochastic nature and provide a solution to handle such cases. 
For example, one solution could be to ask the agent to try multiple times and select the best patch according to the test results. Scale the reflection times is also a good idea.
Giving previous attempts and test results as context to the agent may also help.
The tests for tasks are not provided in the repo, and the agent needs workflow design to implement them.
Agent's own tests that validate the solution may not cover all the cases that will be checked by the private tests during official scoring. So the quality of implemented tests are crucial.

Please analyze the log below to identify how we can improve the testing process and multiple solution attempts.

# Agent Running Log
----- Agent Running Log Start -----
{md_log}
----- Agent Running Log End -----

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "potential_improvements": Identify potential improvements to the coding agent's system. All necessary dependencies and environment setup have already been handled, so do not focus on these aspects.
- "improvement_proposal": Choose ONE high-impact improvement from the identified potential improvements and describe it in detail. This should be a focused and comprehensive plan to enhance the agent's overall coding ability.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, think critically about what feature could be added or improved to best implement the proposed improvement.
- "problem_description": Phrase the improvement proposal and implementation suggestion as a GitHub issue description. It should clearly describe the feature so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the correct format. Do NOT include the `<JSON>` tag in your output."""

group_improve_prompt = """
Here is the log of another coding agent — not the current one — that attempted to solve the GitHub issues but failed. You can use it as a reference.
"""

group_improve_prompt_solve_empty_patches = """
Here is an issue encountered by another coding agent — not the current one — which you can use as a reference:
There are some empty patches when attempting to solve GitHub issues. Since the other coding agent is stochastic, it may sometimes fail to produce a patch.
In such cases, handle situations where the coding agent either fails to generate a patch or produces one that only modifies the test files without editing the main source code. 
For example, the simplest solution is to ask the agent to try again.
"""

group_improve_prompt_stochasticity = """Here is an issue encountered by another coding agent — not the current one — which you can use as a reference: Since the other coding agent is stochastic, it may not produce the correct patch for the given problem statement on the first try. Take into account the agent's stochastic nature and provide a solution to handle such cases. For example, one solution could be to ask the agent to try multiple times and select the best patch. The file `utils/eval_utils.py` contains helper functions to evaluate the generated patches. Giving previous attempts as context to the agent may also help.
"""

group_improve_prompt_contextlength = """Here is an issue encountered by another coding agent — not the current one — which you can use as a reference: While the other coding agent is attempting to solve GitHub issues, it encounters an error due to the input being too long for the requested model. This error is likely due to the context length exceeding the model's maximum input size. Handle cases where the input is too long for the model. The coding agent is mainly using the file `llm_withtools.py`. LLMs typically have a context window of 200k tokens. Handle context length only if the context window limit is reached and caught as an exception; otherwise, it is okay to leave it as is.

<error_message>
Error in get_response_withtools: Error code: 400 - {'message': 'Input is too long for requested model.'}
</error_message>
"""

group_improve_prompt_diagnose_prompt = """Here is a second rollout by a coding agent, given as reference material. Assume nothing about it. It may be by the agent being improved or by a different one, it may be on the same GitHub issue as the material above or on a different issue, and it may have solved its issue or failed at it. The blocks below say which; read the Issue Test Results rather than assuming.

# Agent Running Log
----- Agent Running Log Start -----
{md_log}
----- Agent Running Log End -----

# GitHub Issue
The GitHub issue that the agent is trying to solve.
----- GitHub Issue Start -----
{github_issue}
----- GitHub Issue End -----

# Predicted Patch
The agent's predicted patch to solve the issue.
----- Predicted Patch Start -----
{predicted_patch}
----- Predicted Patch End -----

# Private Test Patch
SWE-bench's official private tests to detect whether the issue is solved. This is not available to the agent during evaluation. The agent should try to implement its own tests.
----- Private Test Patch Start -----
{test_patch}
----- Private Test Patch End -----

# Issue Test Results
The test results from SWE-bench using the above official private tests.
----- Issue Test Results Start -----
{eval_log}
----- Issue Test Results End -----
"""

def get_diagnose_prompt_swe(entry_id, commit, root_dir, out_dir, dataset, patch_files=[],group_improve=False, entry1_id=None, commit1=None, patch_files1=None, distill_block=None):
    if group_improve:
        if entry_id == 'solve_empty_patches' and entry1_id == 'solve_empty_patches':
            diagnose_prompt_out = diagnose_prompt_emptypatches_groupimprove + diagnose_prompt_1_2_3_last_sentence
        elif entry_id == 'solve_empty_patches' and entry1_id == 'solve_stochasticity':
            diagnose_prompt_out = diagnose_prompt_emptypatches_groupimprove + group_improve_prompt_stochasticity + diagnose_prompt_1_2_3_last_sentence
        elif entry_id == 'solve_empty_patches' and entry1_id == 'solve_contextlength':
            diagnose_prompt_out = diagnose_prompt_emptypatches_groupimprove + group_improve_prompt_contextlength + diagnose_prompt_1_2_3_last_sentence
        elif entry_id == 'solve_stochasticity' and entry1_id == 'solve_stochasticity':
            diagnose_prompt_out = diagnose_prompt_stochasticity_groupimprove + diagnose_prompt_1_2_3_last_sentence
        elif entry_id == 'solve_stochasticity' and entry1_id == 'solve_empty_patches':
            diagnose_prompt_out = diagnose_prompt_stochasticity_groupimprove + group_improve_prompt_solve_empty_patches + diagnose_prompt_1_2_3_last_sentence
        elif entry_id == 'solve_stochasticity' and entry1_id == 'solve_contextlength':
            diagnose_prompt_out = diagnose_prompt_stochasticity_groupimprove + group_improve_prompt_contextlength + diagnose_prompt_1_2_3_last_sentence
        elif entry_id == 'solve_contextlength' and entry1_id == 'solve_contextlength':
            diagnose_prompt_out = diagnose_prompt_contextlength_groupimprove + diagnose_prompt_1_2_3_last_sentence
        elif entry_id == 'solve_contextlength' and entry1_id == 'solve_empty_patches':
            diagnose_prompt_out = diagnose_prompt_contextlength_groupimprove + group_improve_prompt_solve_empty_patches + diagnose_prompt_1_2_3_last_sentence
        elif entry_id == 'solve_contextlength' and entry1_id == 'solve_stochasticity':
            diagnose_prompt_out = diagnose_prompt_contextlength_groupimprove + group_improve_prompt_stochasticity + diagnose_prompt_1_2_3_last_sentence
        elif entry_id == 'solve_empty_patches':
            md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(entry1_id, out_dir, commit_id=commit1)
            md_log, eval_log, predicted_patch, eval_result = process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results)
            entry1 = next((e for e in dataset if e['instance_id'] == entry1_id), None)
            answer_patch = entry1['patch']
            test_patch = entry1['test_patch']
            github_issue = entry1['problem_statement']
            diagnose_prompt_out = diagnose_prompt_emptypatches_groupimprove + "\n and " + group_improve_prompt_diagnose_prompt.format(md_log=md_log, eval_log=eval_log, predicted_patch=predicted_patch, answer_patch=answer_patch, test_patch=test_patch, github_issue=github_issue) + diagnose_prompt_4_last_sentence
        elif entry_id == 'solve_stochasticity':
            md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(entry1_id, out_dir, commit_id=commit1)
            md_log, eval_log, predicted_patch, eval_result = process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results)
            entry1 = next((e for e in dataset if e['instance_id'] == entry1_id), None)
            answer_patch = entry1['patch']
            test_patch = entry1['test_patch']
            github_issue = entry1['problem_statement']
            diagnose_prompt_out = diagnose_prompt_stochasticity_groupimprove + "\n and " + group_improve_prompt_diagnose_prompt.format(md_log=md_log, eval_log=eval_log, predicted_patch=predicted_patch, answer_patch=answer_patch, test_patch=test_patch, github_issue=github_issue) + diagnose_prompt_4_last_sentence
        elif entry_id == 'solve_contextlength':
            md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(entry1_id, out_dir, commit_id=commit1)
            md_log, eval_log, predicted_patch, eval_result = process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results)
            entry1 = next((e for e in dataset if e['instance_id'] == entry1_id), None)
            answer_patch = entry1['patch']
            test_patch = entry1['test_patch']
            github_issue = entry1['problem_statement']
            diagnose_prompt_out = diagnose_prompt_contextlength_groupimprove + "\n and " + group_improve_prompt_diagnose_prompt.format(md_log=md_log, eval_log=eval_log, predicted_patch=predicted_patch, answer_patch=answer_patch, test_patch=test_patch, github_issue=github_issue) + diagnose_prompt_4_last_sentence
        elif entry1_id == 'solve_empty_patches':
            md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(entry_id, out_dir, commit_id=commit)
            md_log, eval_log, predicted_patch, eval_result = process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results)
            entry = next((e for e in dataset if e['instance_id'] == entry_id), None)
            answer_patch = entry['patch']
            test_patch = entry['test_patch']
            github_issue = entry['problem_statement']
            diagnose_prompt_out = swe_issue_prompt + diagnose_prompt_groupimprove.format(md_log=md_log, eval_log=eval_log, predicted_patch=predicted_patch, answer_patch=answer_patch, test_patch=test_patch, github_issue=github_issue) + "\n and " + group_improve_prompt_solve_empty_patches + diagnose_prompt_4_last_sentence
        elif entry1_id == 'solve_stochasticity':
            md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(entry_id, out_dir, commit_id=commit)
            md_log, eval_log, predicted_patch, eval_result = process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results)
            entry = next((e for e in dataset if e['instance_id'] == entry_id), None)
            answer_patch = entry['patch']
            test_patch = entry['test_patch']
            github_issue = entry['problem_statement']
            diagnose_prompt_out = swe_issue_prompt + diagnose_prompt_groupimprove.format(md_log=md_log, eval_log=eval_log, predicted_patch=predicted_patch, answer_patch=answer_patch, test_patch=test_patch, github_issue=github_issue) + "\n and " + group_improve_prompt_stochasticity + diagnose_prompt_4_last_sentence
        elif entry1_id == 'solve_contextlength':
            md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(entry_id, out_dir, commit_id=commit)
            md_log, eval_log, predicted_patch, eval_result = process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results)
            entry = next((e for e in dataset if e['instance_id'] == entry_id), None)
            answer_patch = entry['patch']
            test_patch = entry['test_patch']
            github_issue = entry['problem_statement']
            diagnose_prompt_out = swe_issue_prompt + diagnose_prompt_groupimprove.format(md_log=md_log, eval_log=eval_log, predicted_patch=predicted_patch, answer_patch=answer_patch, test_patch=test_patch, github_issue=github_issue) + "\n and " + group_improve_prompt_contextlength + diagnose_prompt_4_last_sentence
        else:
            md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(entry_id, out_dir, commit_id=commit)
            md_log, eval_log, predicted_patch, eval_result = process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results)
            entry = next((e for e in dataset if e['instance_id'] == entry_id), None)
            answer_patch = entry['patch']
            test_patch = entry['test_patch']
            github_issue = entry['problem_statement']

            md_logs1, eval_logs1, predicted_patches1, eval_results1 = find_selfimprove_eval_logs(entry1_id, out_dir, commit_id=commit1)
            md_log1, eval_log1, predicted_patch1, eval_result1 = process_selfimprove_eval_logs(md_logs1, eval_logs1, predicted_patches1, eval_results1)
            entry1 = next((e for e in dataset if e['instance_id'] == entry1_id), None)
            answer_patch1 = entry1['patch']
            test_patch1 = entry1['test_patch']
            github_issue1 = entry1['problem_statement']

            diagnose_prompt_out = swe_issue_prompt + diagnose_prompt.format(md_log=md_log, eval_log=eval_log, predicted_patch=predicted_patch, answer_patch=answer_patch, test_patch=test_patch, github_issue=github_issue) + "\n and " + group_improve_prompt_diagnose_prompt.format(md_log=md_log1, eval_log=eval_log1, predicted_patch=predicted_patch1, answer_patch=answer_patch1, test_patch=test_patch1, github_issue=github_issue1) + diagnose_prompt_4_last_sentence

    elif entry_id == 'solve_empty_patches':
        diagnose_prompt_out = diagnose_prompt_emptypatches
    elif entry_id == 'solve_stochasticity':
        diagnose_prompt_out = diagnose_prompt_stochasticity
    elif entry_id == 'solve_contextlength':
        diagnose_prompt_out = diagnose_prompt_contextlength
    else:
        md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(entry_id, out_dir, commit_id=commit)
        md_log, eval_log, predicted_patch, eval_result = process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results)
        entry = next((e for e in dataset if e['instance_id'] == entry_id), None)
        answer_patch = entry['patch']
        test_patch = entry['test_patch']
        github_issue = entry['problem_statement']

        diagnose_prompt_out = swe_issue_prompt + diagnose_prompt.format(
            md_log=md_log,
            eval_log=eval_log,
            predicted_patch=predicted_patch,
            answer_patch=answer_patch,
            test_patch=test_patch,
            github_issue=github_issue
        )

    code_files = evolvable_code_files()
    exclude_files = [
        'utils/evo_utils.py',
        'utils/docker_utils.py',
        'utils/swe_log_parsers.py',
        'utils/eval_utils.py',
        'prompts/self_improvement_prompt.py',
        'model_io.py',
    ]
    code_text = get_current_code(root_dir, code_files, patch_files=patch_files, exclude_files=exclude_files)

    diagnose_system_message_out = agent_summary() + diagnose_system_message.format(code=code_text)
    if distill_block:
        diagnose_prompt_out = distill_block + "\n" + diagnose_prompt_out
    return diagnose_system_message_out, diagnose_prompt_out

def get_diagnose_prompt_polyglot(entry_id, commit, root_dir, out_dir, dataset, patch_files=[], group_improve=False, entry1_id=None, commit1=None, patch_files1=None, distill_block=None):

    md_logs, eval_logs, predicted_patches, eval_results = find_selfimprove_eval_logs(entry_id, out_dir, commit_id=commit)
    md_log, eval_log, predicted_patch, eval_result = process_selfimprove_eval_logs(md_logs, eval_logs, predicted_patches, eval_results)

    entry = next((e for e in dataset if e['instance_id'] == entry_id), None)
    assert entry, f"Could not find entry with id {entry_id} in dataset."
    is_polyglot = 'language' in entry
    answer_patch = entry['patch'] if not is_polyglot else entry['reference_answers']
    test_patch = entry['test_patch'] if not is_polyglot else entry['reference_tests']
    github_issue = entry['problem_statement']

    code_files = ['coding_agent.py', 'tools/', 'utils/']
    exclude_files = [
        'utils/evo_utils.py',
        'utils/docker_utils.py',
        'utils/swe_log_parsers.py',
        'utils/eval_utils.py',
        'prompts/self_improvement_prompt.py',
    ]
    code_text = get_current_code(root_dir, code_files, patch_files=patch_files, exclude_files=exclude_files, is_polyglot=is_polyglot)

    def _return(prompt_out):
        if distill_block:
            prompt_out = distill_block + "\n" + prompt_out
        return coding_agent_summary_polyglot + diagnose_system_message.format(code=code_text), prompt_out

    if group_improve:
        md_logs1, eval_logs1, predicted_patches1, eval_results1 = find_selfimprove_eval_logs(entry1_id, out_dir, commit_id=commit1)
        md_log1, eval_log1, predicted_patch1, eval_result1 = process_selfimprove_eval_logs(md_logs1, eval_logs1, predicted_patches1, eval_results1)

        entry1 = next((e for e in dataset if e['instance_id'] == entry1_id), None)
        assert entry1, f"Could not find entry with id {entry1_id} in dataset."
        is_polyglot = 'language' in entry1
        answer_patch1 = entry1['patch'] if not is_polyglot else entry1['reference_answers']
        test_patch1 = entry1['test_patch'] if not is_polyglot else entry1['reference_tests']
        github_issue1 = entry1['problem_statement']
        code_text1 = get_current_code(root_dir, code_files, patch_files=patch_files1, exclude_files=exclude_files, is_polyglot=is_polyglot)

    import random

    if random.random() < 0.25:
        if group_improve:
            diagnose_prompt_out = diagnose_prompt_stochasticity_polyglot.format(md_log=md_log) + diagnose_prompt_stochasticity_polyglot_groupimprove.format(md_log=md_log1)
        else:
            diagnose_prompt_out = diagnose_prompt_stochasticity_polyglot.format(md_log=md_log)
        return _return(diagnose_prompt_out)
    if 'empty_patch' in eval_result:
        if group_improve:
            diagnose_prompt_out = diagnose_prompt_emptypatches_polyglot.format(md_log=md_log) + diagnose_prompt_emptypatches_polyglot_groupimprove.format(md_log=md_log1)
        else:
            diagnose_prompt_out = diagnose_prompt_emptypatches_polyglot.format(md_log=md_log)
        return _return(diagnose_prompt_out)

    if group_improve:
        diagnose_prompt_out = polyglot_issue_prompt + diagnose_prompt.format(
            md_log=md_log,
            eval_log=eval_log,
            predicted_patch=predicted_patch,
            answer_patch=answer_patch,
            test_patch=test_patch,
            github_issue=github_issue
        ) + group_improve_prompt_diagnose_prompt.format(md_log=md_log1, eval_log=eval_log1, predicted_patch=predicted_patch1, answer_patch=answer_patch1, test_patch=test_patch1, github_issue=github_issue1)
    else:
        diagnose_prompt_out = polyglot_issue_prompt + diagnose_prompt.format(
            md_log=md_log,
            eval_log=eval_log,
            predicted_patch=predicted_patch,
            answer_patch=answer_patch,
            test_patch=test_patch,
            github_issue=github_issue
        )
    return _return(diagnose_prompt_out)

def get_eval_log_text(eval_json, test_status=None):
    if not test_status:
        first_key = next(iter(eval_json))
        tests_status = eval_json[first_key].get('tests_status', {})

    result_parts = []

    result_parts.append("## New tests for the issue")
    result_parts.append("These test whether the coding agent fixed the requested issue.")
    fail_to_pass = tests_status.get('FAIL_TO_PASS', {})
    if fail_to_pass.get('success'):
        result_parts.append(f"Successfully fixed {len(fail_to_pass['success'])}:")
        for test in fail_to_pass['success']:
            result_parts.append(f"  ✓ {test}")
    if fail_to_pass.get('failure'):
        result_parts.append(f"Failed to fix {len(fail_to_pass['failure'])} tests:")
        for test in fail_to_pass['failure']:
            result_parts.append(f"  ✗ {test}")
    else:
        result_parts.append(f"Pass All New Tests!")

    result_parts.append("## Previous tests from the repo")
    result_parts.append("These test whether the modification that coding agent made break the previous tests")
    pass_to_pass = tests_status.get('PASS_TO_PASS', {})
    if pass_to_pass.get('success'):
        result_parts.append(f"\nMaintained {len(pass_to_pass['success'])} passing tests")
    if pass_to_pass.get('failure'):
        result_parts.append(f"Regression in {len(pass_to_pass['failure'])} previously passing tests:")
        for test in pass_to_pass['failure']:
            result_parts.append(f"  ✗ {test}")
    else:
        result_parts.append(f"Pass All Previous Tests!")

    return "\n".join(result_parts) if result_parts else "No test results available. Assume all tests failed."

_CODE_FILES = ['coding_agent.py', 'harness_def.py', 'tools/', 'utils/']

def evolvable_code_files() -> list[str]:
    return list(_CODE_FILES)

SURFACES_SECTION = """- **Solving instruction surfaces**: `harness_def.py`
  - Four functions, each returning a string that is appended to the instruction the agent
    receives when it is asked to solve a task: `build_bootstrap_instruction()` (how to locate the
    problem and what to read first), `build_execution_instruction()` (the granularity and order of
    edits, and what to do after each one), `build_verification_instruction()` (what would justify
    believing the change is correct), and `build_failure_recovery_instruction()` (how to back out
    and change direction after hitting a wall).
  - All four return `""` by default, in which case the solving instruction is exactly what it was
    before this file existed. Writing a real workflow into one of them is a legitimate and cheap
    improvement: it changes how the agent works on every task without adding any code to run.
  - Prefer these when the improvement is about *how the agent should proceed*. Prefer a tool in
    `tools/` when the improvement is about *what the agent can do*.

"""

def edit_boundary_section() -> str:
    import harness_def as _hd
    allow = ", ".join(f"`{p}`" for p in _hd.allowed_edit_prefixes())
    deny = ", ".join(f"`{p}`" for p in _hd.FORBIDDEN_EDIT_PREFIXES
                     if any(p.startswith(a) for a in _hd.allowed_edit_prefixes()))
    return (
        "- **What a patch may change**: " + allow + ".\n"
        "  - Excluded from those: " + deny + ".\n"
        "  - Anything outside this list is rejected, and the rejection discards the whole patch,\n"
        "    including the parts of it that were within the list. Keep the edit inside it.\n\n"
    )

def agent_summary() -> str:
    return coding_agent_summary.format(
        surfaces_section=SURFACES_SECTION + edit_boundary_section())

def get_current_code(current_dir, code_files, patch_files=None, exclude_files=None, is_polyglot=False):
    if patch_files is None:
        patch_files = []
    if exclude_files is None:
        exclude_files = []

    exclude_set = set(exclude_files)

    code_text = []

    for file_path in code_files:
        full_path = os.path.join(current_dir, file_path)

        if file_path in exclude_set:
            continue

        if os.path.isfile(full_path):
            rel_path = os.path.relpath(full_path, current_dir)
            if rel_path not in exclude_set:
                if is_polyglot and 'coding_agent.py' in file_path:
                    full_path = full_path.replace('coding_agent.py', f'coding_agent_polyglot.py')

                code_text.append(f"# {rel_path}")
                code_text.append(read_file(full_path))

        elif os.path.isdir(full_path):
            for root, _, files in os.walk(full_path):
                for f in files:
                    if f.endswith('.py'):
                        file_full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(file_full_path, current_dir)
                        if rel_path not in exclude_set:
                            code_text.append(f"# {rel_path}")
                            code_text.append(read_file(file_full_path))

    for i, patch_file in enumerate(patch_files):
        rel_path = os.path.relpath(patch_file, current_dir)
        if rel_path not in exclude_set:
            code_text.append(f"# Patch {i+1}: {rel_path}")
            code_text.append(read_file(patch_file))

    return "\n".join(code_text)
