import asyncio
import os
import signal

def tool_info():
    return {
        "name": "bash",
        "description": """Run commands in a bash shell\n
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.\n
* You don't have access to the internet via this tool.\n
* You do have access to a mirror of common linux and python packages via apt and pip.\n
* State is persistent across command calls and discussions with the user.\n
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.\n
* Please avoid commands that may produce a very large amount of output.\n
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run."
                }
            },
            "required": ["command"]
        }
    }

class BashSession:
    def __init__(self):
        self._started = False
        self._process = None
        self._timed_out = False
        self._timeout = 120.0
        self._sentinel = "<<exit>>"
        self._output_delay = 0.2

    async def start(self):
        if self._started:
            return
        self._process = await asyncio.create_subprocess_shell(
            "/bin/bash -i",
            preexec_fn=os.setsid,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy()
        )
        self._started = True

    def stop(self):
        if not self._started:
            return
        if self._process.returncode is None:
            self._process.terminate()
        self._process = None
        self._started = False

    async def run(self, command):
        if not self._started:
            raise ValueError("Session has not started.")
        if self._process.returncode is not None:
            raise ValueError(f"Bash has exited with returncode {self._process.returncode}")
        if self._timed_out:
            raise ValueError(
                f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
            )

        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        try:
            output = ''
            start_time = asyncio.get_event_loop().time()

            while True:
                if asyncio.get_event_loop().time() - start_time > self._timeout:
                    self._timed_out = True
                    raise ValueError(
                        f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
                    )

                await asyncio.sleep(self._output_delay)
                stdout_data = self._process.stdout._buffer.decode(errors='ignore')
                stderr_data = self._process.stderr._buffer.decode(errors='ignore')

                if self._sentinel in stdout_data:
                    output = stdout_data[: stdout_data.index(self._sentinel)]
                    break

            self._process.stdout._buffer.clear()
            self._process.stderr._buffer.clear()

            output = output.strip()
            error = stderr_data.strip()

            return output, error

        except Exception as e:
            self._timed_out = True
            raise ValueError(str(e))

def filter_error(error):
    filtered_lines = []
    i = 0
    error_lines = error.splitlines()
    while i < len(error_lines):
        line = error_lines[i]

        if "Inappropriate ioctl for device" in line:
            i += 3
            if '<<exit>>' in error_lines[i]:
                i += 1
            while i < len(error_lines) - 1:
                filtered_lines.append(error_lines[i])
                i += 1
            i += 1
            continue

        filtered_lines.append(line)
        i += 1
    return '\n'.join(filtered_lines).strip()

async def tool_function_call(command):
    bash_session = BashSession()
    try:
        if not bash_session._started:
            await bash_session.start()

        output, error = await bash_session.run(command)
        error = filter_error(error)
        result = ""
        if output:
            result += output
        if error:
            result += "\nError:\n" + error
        return result.strip()
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        proc = bash_session._process
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            try:
                await proc.wait()
            except Exception:
                pass
        bash_session._process = None
        bash_session._started = False

def tool_function(command):
    return asyncio.run(tool_function_call(command))

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bash.py '<command>'")
    else:
        input_command = ' '.join(sys.argv[1:])
        result = tool_function(input_command)
        print(result)
