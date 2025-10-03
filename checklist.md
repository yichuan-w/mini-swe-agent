TODO Checklist  
Many of these core abstractions are provided in the codebase. Search for TODO for what you need to implement within the codebase! 
Core Architecture Requirement 
Implement a Python class named ReactAgent with attributes and methods
Attribute 
name: str
timestamp: int
id_to_message: List[Dict[str, Any]] # mapping from message ids to  message dictionaries. You cannot remove messages from the list.
Root_message_id
Current_message_id
function_map: Dict[str, Callable] 
def__init__(name: str, response_parser)
def add_functions(tools:: List[Callable])
def set_user_prompt(user_prompt: str)  
def get_instructions(): str
def set_instructions(instructions: str)
def add_message(role: str, content: str): int #returns the message id 
def save_history(file_name: str) # save the attributes as an YAML file 
def run(): str
For SWE tasks, create a customized 
You need to define a list of tools that the agent could use.  You could create an Environment class and describe the tools as the methods of that class.  The Environment class could maintain a state, such as cwd.
You need to instantiate a ReactAgent with the task at hand, add functions, and run the agent. 
Response Parser Implementation 
Create ResponseParser class for custom text format (NOT XML/JSON) 
Parse function calls using format 

----BEGIN_FUNCTION_CALL----
function_name
----ARG----
arg_name # single line
arg_value # multiline
...
---ARG---
...
----END_FUNCTION_CALL----


LLM Class Implementation
Create the LLM abstract class
Create a subclass for O4-mini (med) and implement its generate method.
Tools 
Required tools 
Reuse tools such as run_bash_cmd(command: str, description: str) and finish(result: str) 
Implement add_instructions_and_backtrack(instructions: str, at_message_id: str)  
Enhanced tools (for better accuracy) 
Create replace_in_file(file_path: str, from_line: str, to_line: str, content: str) 
Create show_file(file_path: str)– show file with line numbers 
Add other custom tools based on common LLM mistakes 

Evaluation Setups 
Configure GPT-5-mini (medium reasoning) as the backend LLM 
Create baseline evaluation (without backtracking)
Measure and report accuracy improvements with backtracking and custom tools
If you are brave enough, you can create multiple ReactAgents with different roles, such as Root cause analyzer, Patcher, and Verifier, and use them as a Multi-agent system.  Note that each ReAct agent can be treated as a function whose run method returns the result of running the agent.



