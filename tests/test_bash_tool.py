import pytest
from tools.bash import tool_function, BashSession

@pytest.fixture
def bash_session():
    session = BashSession()
    return session

class TestBashTool:
    def test_simple_command(self):
        result = tool_function("echo 'hello world'")
        assert "hello world" in result
        assert "Error" not in result

    def test_multiple_commands(self):
        result = tool_function("echo 'first' && echo 'second'")
        assert "first" in result
        assert "second" in result

    def test_command_with_error(self):
        result = tool_function("ls /nonexistent/directory")
        assert "Error" in result
        assert "No such file or directory" in result

    def test_environment_variables(self):
        result = tool_function("TEST_VAR='hello' && echo $TEST_VAR")
        assert "hello" in result

    def test_command_output_processing(self):
        commands = [
            "echo 'line1'",
            "echo 'line2'",
            "echo 'line3'"
        ]
        result = tool_function(" && ".join(commands))
        assert all(f"line{i}" in result for i in range(1, 4))

    def test_long_running_command(self):
        result = tool_function("sleep 1 && echo 'done'")
        assert "done" in result

    @pytest.mark.parametrize("invalid_command", [
        "invalid_command_name",
        "cd /nonexistent/path",
        "/bin/nonexistent"
    ])
    def test_invalid_commands(self, invalid_command):
        result = tool_function(invalid_command)
        assert "Error" in result or "command not found" in result

    def test_command_with_special_chars(self):
        result = tool_function("echo 'test with spaces and !@#$%^&*()'")
        assert "test with spaces" in result
        assert "!@#$%^&*()" in result

    def test_multiple_line_output(self):
        command = """printf 'line1\nline2\nline3'"""
        result = tool_function(command)
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_large_output_handling(self):
        command = "for i in {1..100}; do echo \"Line $i\"; done"
        result = tool_function(command)
        assert "Line 1" in result
        assert "Line 100" in result
