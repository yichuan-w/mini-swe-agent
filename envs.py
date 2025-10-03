from utils import get_sb_environment
import subprocess

class LimitsExceeded(Exception):
    """Raised when the agent has reached its step limit."""


class SWEEnvironment:
    """
    Minimal interface to the SWEBench execution environment.

    Students may use their own wrapper. The environment must expose:
    - execute(command: str) -> str: Run a shell command and return stdout, or raise ValueError on failure
    """

    def __init__(self, instance: dict):
        self.env = get_sb_environment(instance)
    
    def _to_text(self, result) -> str:
        """Normalize environment execute() result to text."""
        if isinstance(result, str):
            return result
        if isinstance(result, bytes):
            return result.decode("utf-8", errors="replace")
        if isinstance(result, dict):
            # Prefer a combined 'output' if present
            combined = result.get("output", None)
            if combined is not None:
                if isinstance(combined, bytes):
                    combined = combined.decode("utf-8", errors="replace")
                return str(combined)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return f"--STDOUT--\n{stdout}\n--STDERR--\n{stderr}"
        return str(result)
     
    # -------------------- REQUIRED TOOLS --------------------
    def run_bash_cmd(self, command: str) -> str:
        """
        Run the command in a bash shell and return the output or throw a ValueError
        if the process returns non-zero exit code.

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        try:
            output = self.env.execute(command)
            output = self._to_text(output)
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ValueError(output)
        except TimeoutError:
            raise ValueError("TimeoutError")
        return output
    
    def generate_patch(self, result: str) -> str:
        """
        Generate a patch from the result (for SWE-Bench)
        """
        try:
            # Debug: show working tree and staged files
            try:
                pre_status = self.env.execute("git status --porcelain")
                print("[DEBUG] git status --porcelain:\n", self._to_text(pre_status))
            except Exception as _:
                pass

            # Stage all changes
            try:
                add_out = self.env.execute("git add -A")
                add_out_txt = self._to_text(add_out)
                if add_out_txt.strip():
                    print("[DEBUG] git add -A output:\n", add_out_txt)
            except Exception as _:
                pass

            try:
                name_only = self.env.execute("git diff --cached --name-only")
                print("[DEBUG] git diff --cached --name-only:\n", self._to_text(name_only))
            except Exception as _:
                pass

            patch_output = self.env.execute("git diff --cached")
            # Extract only unified diff from env result. Do NOT include any narrative.
            if isinstance(patch_output, dict):
                out_val = patch_output.get("output", None)
                if out_val is None:
                    out_val = patch_output.get("stdout", b"")
                if isinstance(out_val, bytes):
                    out_val = out_val.decode("utf-8", errors="replace")
                patch_text = out_val if isinstance(out_val, str) else str(out_val)
            else:
                patch_text = self._to_text(patch_output)

            # If empty or whitespace, submit an empty patch
            if not patch_text or not patch_text.strip():
                print("[DEBUG] No staged diff detected (empty patch).")
                return ""

            # Ensure trailing newline to avoid 'ends in middle of line'
            if not patch_text.endswith("\n"):
                patch_text += "\n"
            # Debug preview (first 500 chars)
            print("[DEBUG] Unified diff preview (first 500 chars):\n", patch_text[:500])
            return patch_text
        except Exception as e:
            # On error, return empty patch; the harness will mark it accordingly
            print("[DEBUG] Error generating patch:", e)
            return ""
    
    # -------------------- TODO(student): add more functions here if you want --------------------
    def has_staged_changes(self) -> bool:
        """Return True if there are staged changes in the repo (cached diff non-empty)."""
        try:
            out = self.env.execute("git diff --cached --name-only")
            txt = self._to_text(out)
            return bool(txt.strip())
        except Exception:
            return False

    # -------------------- Optional: Leann code search helpers --------------------
    def leann_install(self) -> str:
        """Install leann (and dependencies) inside the sandbox."""
        try:
            return self._to_text(self.env.execute("python -m pip install -q leann"))
        except Exception as e:
            raise ValueError(str(e))

    def leann_build_index(self, index_name: str = "code-base-index", docs_glob: str = "**/*.py") -> str:
        """Build a leann index over code files using sentence-transformers + hnsw backend."""
        try:
            cmd = (
                "leann build " + index_name +
                " --docs $(git ls-files)" +
                " --embedding-mode sentence-transformers"
                " --embedding-model all-MiniLM-L6-v2"
                " --backend hnsw --force --no-recompute"
            )
            return self._to_text(self.env.execute(cmd))
        except Exception as e:
            raise ValueError(str(e))

    def leann_search(self, index_name: str, query: str, k: int = 5) -> str:
        """Search the leann index for semantic matches to a query."""
        try:
            cmd = f"leann search {index_name} \"{query}\" --k {k}"
            return self._to_text(self.env.execute(cmd))
        except Exception as e:
            raise ValueError(str(e))
    def replace_in_file(self, file_path: str, from_line: int, to_line: int, content: str) -> str:
        """
        [Optional] Replace the content of the file from the given line to the given line with the given content
        """
        if from_line <= 0 or to_line < from_line:
            raise ValueError("Invalid line range")
        # Use ed or sed to modify lines safely; here we construct a temporary file with awk
        awk_cmd = (
            "awk 'NR<" + str(from_line) + "{print;next} NR>" + str(to_line) + "{p=1} p{print} ' "
            + file_path + " > "+ file_path + ".bak"
        )
        try:
            # Create backup with lines outside the range
            self._to_text(self.env.execute(awk_cmd))
            # Insert new content at from_line position
            # Build a here-doc to insert content
            heredoc = (
                "ed -s " + file_path + 
                " <<'EDCMDS'\n" +
                str(from_line) + "," + str(to_line) + "d\n" +
                str(from_line - 1) + "a\n" + content + "\n.\n" +
                "w\nq\nEDCMDS"
            )
            self._to_text(self.env.execute(heredoc))
            return "Replaced lines " + str(from_line) + "-" + str(to_line) + " in " + file_path
        except Exception as e:
            raise ValueError(str(e))
    
    def show_file(self, file_path: str) -> str:
        """
        [Optional]Show the content of the file
        """
        try:
            return self._to_text(self.env.execute(f"nl -ba -- {file_path}"))
        except Exception as e:
            raise ValueError(str(e))

class DumbEnvironment:
    """
    Dumb environment that just executes the command
    """

    def execute(self, command: str) -> str:
        """
        Run the command in bash and return the output

        Args;
            command (str): the shell command to run

        Returns:
            The output of running the shell command
        """
        result = subprocess.run(command, capture_output=True, shell=True, check=False)
        output = f"--STDOUT--\n{result.stdout.decode()}\n--STDERR--\n{result.stderr.decode()}"
        if result.returncode:
            raise ValueError(output)
        return output