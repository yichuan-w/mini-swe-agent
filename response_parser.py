class ResponseParser:
    """
    Parses LLM responses to extract a single function call using a rigid textual format.

    The LLM must output exactly one function call at the end of its response.
    Do NOT use JSON or XML. Use rfind to locate the final markers.
    """

    BEGIN_CALL = "----BEGIN_FUNCTION_CALL----"
    END_CALL = "----END_FUNCTION_CALL----"
    ARG_SEP = "----ARG----"

    # Students should include this exact template in the system prompt so the LLM follows it.
    response_format = f"""
your_thoughts_here
...
{BEGIN_CALL}
function_name
{ARG_SEP}
arg1_name
arg1_value (can be multiline)
{ARG_SEP}
arg2_name
arg2_value (can be multiline)
...
{END_CALL}
"""

    def parse(self, text: str) -> dict:
        """
        Parse the function call from `text` using string.rfind to avoid confusion with
        earlier delimiter-like content in the reasoning.

        Returns a dictionary: {"thought": str, "name": str, "arguments": dict}
        """
        end_idx = text.rfind(self.END_CALL)
        if end_idx == -1:
            raise ValueError("Missing END_CALL marker in response")

        begin_idx = text.rfind(self.BEGIN_CALL, 0, end_idx)
        if begin_idx == -1:
            raise ValueError("Missing BEGIN_CALL marker before END_CALL in response")

        thought = text[:begin_idx].rstrip()
        call_block = text[begin_idx + len(self.BEGIN_CALL):end_idx]

        # Normalize newlines and strip outer whitespace
        call_block = call_block.strip("\n")
        if not call_block:
            raise ValueError("Empty function call block")

        # Split by ARG_SEP. First segment is function name, following come in pairs (name, value)
        segments = call_block.split(self.ARG_SEP)
        segments = [seg.strip("\n") for seg in segments]
        if len(segments) < 1:
            raise ValueError("Malformed function call: no segments found")

        # Function name is the first non-empty line of the first segment
        func_name_block = segments[0].strip()
        func_name_lines = [ln for ln in func_name_block.splitlines() if ln.strip()]
        if not func_name_lines:
            raise ValueError("Malformed function call: missing function name")
        func_name = func_name_lines[0].strip()

        # Parse arguments: subsequent segments should be name then value
        arguments: dict = {}
        # After splitting, segments[1:] contains alternating arg_name, arg_value blocks
        # But format shows each ARG_SEP introduces a name and then value until next sep.
        # We'll iterate by reading name from the first line of the segment and value from the remaining
        pending_name = None
        pending_value_lines: list[str] = []

        # We treat each segment after the first as a pair block (name on first line, rest value)
        for seg in segments[1:]:
            seg_lines = seg.splitlines()
            if not seg_lines:
                continue
            arg_name = seg_lines[0].strip()
            arg_value = "\n".join(seg_lines[1:]) if len(seg_lines) > 1 else ""
            if not arg_name:
                raise ValueError("Malformed function call: empty argument name")
            # If argument name repeats, last one wins
            arguments[arg_name] = arg_value

        return {"thought": thought, "name": func_name, "arguments": arguments}
