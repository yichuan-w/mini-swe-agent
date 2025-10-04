"""
Starter scaffold for the CS 294-264 HW1 ReAct agent.

Students must implement a minimal ReAct agent that:
- Maintains a message history tree (role, content, timestamp, unique_id, parent, children)
- Uses a textual function-call format (see ResponseParser) with rfind-based parsing
- Alternates Reasoning and Acting until calling the tool `finish`
- Supports tools: `run_bash_cmd`, `finish`, and `add_instructions_and_backtrack`

This file intentionally omits core implementations and replaces them with
clear specifications and TODOs.
"""

from typing import List, Callable, Dict, Any

from response_parser import ResponseParser
from llm import LLM, OpenAIModel
import inspect

class ReactAgent:
    """
    Minimal ReAct agent that:
    - Maintains a message history tree with unique ids
    - Builds the LLM context from the root to current node
    - Registers callable tools with auto-generated docstrings in the system prompt
    - Runs a Reason-Act loop until `finish` is called or MAX_STEPS is reached
    """

    def __init__(self, name: str, parser: ResponseParser, llm: LLM):
        self.name: str = name
        self.parser = parser
        self.llm = llm

        # Message tree storage
        self.id_to_message: List[Dict[str, Any]] = []
        self.root_message_id: int = -1
        self.current_message_id: int = -1

        # Registered tools
        self.function_map: Dict[str, Callable] = {}

        # Set up the initial structure of the history
        # Create required root nodes and a user node (task) and an instruction node.
        self.system_message_id = self.add_message(
            "system",
            (
                "You are a Smart ReAct agent for SWE-Bench. Obey the protocol strictly, think step-by-step, and make real code changes.\n"
                "- Always reason briefly, then ACT using exactly one tool call.\n"
                "- Prefer small, verifiable steps. Read tool output and adapt.\n"
                "- Do not finish until changes are staged (git diff --cached --name-only non-empty).\n"
                "- Output must follow the function-call format with the end token.\n"
            ),
        )
        self.user_message_id = self.add_message("user", "")
        self.instructions_message_id = self.add_message(
            "instructor",
            (
                "SWE-Bench Playbook — YOU MUST PRODUCE CODE CHANGES. Single tool: run_bash_cmd. Think hard, observe feedback, execute precise commands, verify, then finish.\n\n"
                "NON-NEGOTIABLE RULES:\n"
                "- Do NOT call finish until there are staged files (git diff --cached --name-only is non-empty).\n"
                "- Every message MUST end with EXACTLY ONE function call in the specified format and the end token.\n"
                "- Use ONLY run_bash_cmd (plus finish/add_instructions_and_backtrack).\n"
                "- No interactive editors. Use ed or a small python here-doc for edits.\n"
                "- Keep outputs minimal (use -q, head -n, tail -n, or sed -n).\n\n"
                "WORKFLOW (REPEAT UNTIL PASS):\n"
                "1) Test: run_bash_cmd('pytest -q').\n"
                "2) Triage: grep for symbols or failing tests: run_bash_cmd('grep -R \"<pattern>\" -n .'); open files: run_bash_cmd('nl -ba path/to/file.py | sed -n \"<start>,<end>p\"').\n"
                "3) Edit (precise, minimal):\n"
                "   - ed example (replace range):\n"
                "     ed -s path/to/file.py <<'ED'\n<from>,<to>d\n<from-1>a\n<new code here>\n.\nw\nq\nED\n"
                "   - python in-place example:\n"
                "     python - <<'PY'\nfrom pathlib import Path\np = Path('path/to/file.py')\ns = p.read_text()\ns = s.replace('OLD', 'NEW', 1)\np.write_text(s)\nPY\n"
                "4) Re-test: run_bash_cmd('pytest -q'); move to next failure.\n"
                "5) Stage+verify: run_bash_cmd('git add -A && git diff --cached --name-only'). Must list files.\n"
                "6) Finish ONLY then.\n\n"
                "LEANN Semantic Search (Required):\n"
                "- The code index 'code-base-index' is prebuilt. Do NOT rebuild it.\n"
                "- Use leann search when exact grep is brittle or you need concept-level matches (APIs, behaviors, responsibilities) across files.\n"
                "- Typical triggers: failing tests mention symbols not found by grep; need similar implementations; locate call sites by meaning.\n\n"
                "STRATEGY & FEEDBACK:\n"
                "- After each command, read output and plan the next exact command (file, lines, change).\n"
                "- If a command fails, fix the command and retry; do not give up.\n"
                "- If stuck after two loops, use add_instructions_and_backtrack to refine the plan.\n\n"
                "EXAMPLES:\n"
                "Run tests:\n"
                "----BEGIN_FUNCTION_CALL----\nrun_bash_cmd\n----ARG----\ncommand\npytest -q\n----END_FUNCTION_CALL----\n\n"
                "Search code with leann (index is ready):\n"
                "----BEGIN_FUNCTION_CALL----\nleann search\n----ARG----\ncode-base-index\n----ARG----\nquery\nwhere is ReactAgent implemented\n----ARG----\nk\n5\n----END_FUNCTION_CALL----\n\n"
                "One-shot with leann search: When a failure references 'Collector.can_fast_delete' but path is unknown, prefer semantic search:\n"
                "----BEGIN_FUNCTION_CALL----\nleann search\n----ARG----\ncode-base-index\n----ARG----\nquery\nimplementations of Collector.can_fast_delete or deletion behavior logic\n----ARG----\nk\n5\n----END_FUNCTION_CALL----\n\n"
                "View lines 110-160:\n"
                "----BEGIN_FUNCTION_CALL----\nrun_bash_cmd\n----ARG----\ncommand\nnl -ba path/to/file.py | sed -n '110,160p'\n----END_FUNCTION_CALL----\n\n"
                "Apply edit via ed:\n"
                "----BEGIN_FUNCTION_CALL----\nrun_bash_cmd\n----ARG----\ncommand\ned -s path/to/file.py <<'ED'\n120,135d\n119a\n<new code here>\n.\nw\nq\nED\n----END_FUNCTION_CALL----\n\n"
                "Stage and verify:\n"
                "----BEGIN_FUNCTION_CALL----\nrun_bash_cmd\n----ARG----\ncommand\ngit add -A && git diff --cached --name-only\n----END_FUNCTION_CALL----\n\n"
                "Finish (only after staged files exist):\n"
                "----BEGIN_FUNCTION_CALL----\nfinish\n----ARG----\nresult\nApplied minimal fix; tests pass locally.\n----END_FUNCTION_CALL----\n"
            ),
        )
        
        # NOTE: mandatory finish function that terminates the agent
        self.add_functions([self.finish])

    # -------------------- MESSAGE TREE --------------------
    def add_message(self, role: str, content: str) -> int:
        """
        Create a new message and add it to the tree.

        The message must include fields: role, content, timestamp, unique_id, parent, children.
        Maintain a pointer to the current node and the root node.
        """
        unique_id = len(self.id_to_message) + 1
        parent_id = self.current_message_id if self.current_message_id != -1 else None
        message = {
            "role": role,
            "content": content,
            "timestamp": unique_id,  # simple monotonic timestamp surrogate
            "unique_id": unique_id,
            "parent": parent_id,
            "children": [],
        }
        self.id_to_message.append(message)

        # Link into parent's children if exists
        if parent_id is not None:
            self.id_to_message[parent_id - 1]["children"].append(unique_id)

        # Initialize root and current pointers
        if self.root_message_id == -1:
            self.root_message_id = unique_id
        self.current_message_id = unique_id
        return unique_id

    def set_message_content(self, message_id: int, content: str) -> None:
        """Update message content by id."""
        if message_id <= 0 or message_id > len(self.id_to_message):
            raise ValueError("Invalid message_id")
        self.id_to_message[message_id - 1]["content"] = content

    def get_context(self) -> str:
        """
        Build the full LLM context by walking from the root to the current message.
        """
        # Build path from root to current by following parents
        if self.current_message_id == -1:
            return ""
        path: list[int] = []
        cursor = self.current_message_id
        while cursor is not None:
            path.append(cursor)
            parent = self.id_to_message[cursor - 1]["parent"]
            cursor = parent
        path.reverse()

        # Concatenate each message context block
        parts: list[str] = []
        for mid in path:
            parts.append(self.message_id_to_context(mid))
        return "".join(parts)

    # -------------------- REQUIRED TOOLS --------------------
    def add_functions(self, tools: List[Callable]):
        """
        Add callable tools to the agent's function map.

        The system prompt must include tool descriptions that cover:
        - The signature of each tool
        - The docstring of each tool
        """
        for tool in tools:
            self.function_map[tool.__name__] = tool
        # Ensure core tools are present
        self.function_map["finish"] = self.finish
        self.function_map["add_instructions_and_backtrack"] = self.add_instructions_and_backtrack
        # Keep system message minimal; the full tool list is rendered dynamically in message_id_to_context
        base = (
            "You are a Smart ReAct agent for SWE-Bench. Obey the protocol strictly, think step-by-step, and make real code changes.\n"
            "- Always reason briefly, then ACT using exactly one tool call.\n"
            "- Prefer small, verifiable steps. Read tool output and adapt.\n"
            "- Do not finish until changes are staged (git diff --cached --name-only non-empty).\n"
            "- Output must follow the function-call format with the end token.\n"
        )
        self.set_message_content(self.system_message_id, base)
    
    def finish(self, result: str):
        """The agent must call this function with the final result when it has solved the given task. The function calls "git add -A and git diff --cached" to generate a patch and returns the patch as submission.

        Args: 
            result (str); the result generated by the agent

        Returns:
            The result passed as an argument.  The result is then returned by the agent's run method.
        """
        return result 

    def add_instructions_and_backtrack(self, instructions: str, at_message_id: int):
        """
        The agent should call this function if it is making too many mistakes or is stuck.

        The function changes the content of the instruction node with 'instructions' and
        backtracks at the node with id 'at_message_id'. Backtracking means the current node
        pointer moves to the specified node and subsequent context is rebuilt from there.

        Returns a short success string.
        """
        # Update instruction node content
        self.set_message_content(self.instructions_message_id, instructions)
        # Validate target id and backtrack current pointer
        if at_message_id <= 0 or at_message_id > len(self.id_to_message):
            raise ValueError("Invalid backtrack message id")
        self.current_message_id = at_message_id
        return "Updated instructions and backtracked"

    # -------------------- MAIN LOOP --------------------
    def run(self, task: str, max_steps: int) -> str:
        """
        Run the agent's main ReAct loop:
        - Set the user prompt
        - Loop up to max_steps (<= 100):
            - Build context from the message tree
            - Query the LLM
            - Parse a single function call at the end (see ResponseParser)
            - Execute the tool
            - Append tool result to the tree
            - If `finish` is called, return the final result
        """
        if max_steps > 100:
            max_steps = 100
        # Set user task content into the user node, and position current at instructor
        self.set_message_content(self.user_message_id, task)
        self.current_message_id = self.instructions_message_id

        for step in range(max_steps):
            print(f"Step {step}")
            # Build context and query LLM
            context = self.get_context()
            try:
                llm_output = self.llm.generate(context)
            except Exception as e:
                # Record error as assistant message and continue
                self.add_message("assistant", f"LLM error: {e}")
                continue

            # Add assistant raw output
            assistant_id = self.add_message("assistant", llm_output)

            # Parse function call
            try:
                call = self.parser.parse(llm_output)
            except Exception as e:
                # Add tool error and continue
                self.add_message("tool", f"Parser error: {e}")
                continue

            name = call.get("name")
            args = call.get("arguments", {})
            if name not in self.function_map:
                self.add_message("tool", f"Unknown function: {name}")
                continue

            tool = self.function_map[name]
            # Execute tool
            try:
                result = tool(**args)
            except TypeError:
                # Fallback: try passing a single positional dict if signature mismatches
                try:
                    result = tool(args)
                except Exception as e:
                    result = f"Tool execution error: {e}"
            except Exception as e:
                result = f"Tool execution error: {e}"

            # Append tool result
            self.add_message("tool", str(result))
            print(f"Tool result: {result}")

            # If finish was called, return result as final output
            if name == "finish":
                print(f"Finish called with result: {result}")
                return str(result)

        # If we exit the loop without finish, return a default message
        return "MAX_STEPS reached without calling finish"

    def message_id_to_context(self, message_id: int) -> str:
        """
        Helper function to convert a message id to a context string.
        """
        # message_id is 1-based; list index is 0-based
        message = self.id_to_message[message_id - 1]
        header = f'----------------------------\n|MESSAGE(role="{message["role"]}", id={message["unique_id"]})|\n'
        content = message["content"]
        if message["role"] == "system":
            tool_descriptions = []
            for tool in self.function_map.values():
                signature = inspect.signature(tool)
                docstring = inspect.getdoc(tool)
                tool_description = f"Function: {tool.__name__}{signature}\n{docstring}\n"
                tool_descriptions.append(tool_description)

            tool_descriptions = "\n".join(tool_descriptions)
            return (
                f"{header}{content}\n"
                f"--- AVAILABLE TOOLS ---\n{tool_descriptions}\n\n"
                f"--- RESPONSE FORMAT ---\n{self.parser.response_format}\n"
            )
        elif message["role"] == "instructor":
            return f"{header}YOU MUST FOLLOW THE FOLLOWING INSTRUCTIONS AT ANY COST. OTHERWISE, YOU WILL BE DECOMISSIONED.\n{content}\n"
        else:
            return f"{header}{content}\n"

def main():
    from envs import DumbEnvironment
    llm = OpenAIModel("----END_FUNCTION_CALL----", "gpt-4o-mini")
    parser = ResponseParser()

    env = DumbEnvironment()
    dumb_agent = ReactAgent("dumb-agent", parser, llm)
    dumb_agent.add_functions([env.run_bash_cmd])
    result = dumb_agent.run("Show the contents of all files in the current directory.", max_steps=10)
    print(result)

if __name__ == "__main__":
    # Optional: students can add their own quick manual test here.
    main()