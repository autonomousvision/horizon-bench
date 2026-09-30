from ai_researcher.classes.AbstractTool import AbstractTool
import os
import subprocess
from ai_researcher import Config

class CodeStructureTool(AbstractTool):
    name: str = "gen_code_tree_structure"
    description: str = """
Generate a tree structure of the code in the specified directory. Use this function when you need to know the overview of the codebase and want to generate a tree structure of the codebase.

Args:
    directory: The directory to generate the tree structure for.
Returns:
    A string representation of the tree structure of the code in the specified directory.
"""
    parameters: dict[str, dict[str, str]] = {
        "directory": {
            "type": "string",
            "description": "The directory to generate the tree structure for."
        }
    }

    def __call__(self, directory: str, *args, **kwargs):    
        try:
            command = f"tree {directory}"
            response = subprocess.check_output(command, shell=True, text=True)
            return response
        except Exception as e:
            return f"Error running tree {directory}: {str(e)}"
        

class ReadFileTool(AbstractTool):
    name: str = "read_file"
    description: str = """
Read the contents of a file and return it as a string. Use this function when there is a need to check an existing file.
Args:
    file_path: The path of the file to read.
Returns:
    A string representation of the contents of the file.
"""
    parameters: dict[str, dict[str, str]] = {
        "file_path": {
            "type": "string",
            "description": "The path of the file to read."
        }
    }

    def __call__(self, file_path: str, *args, **kwargs):    
        try:
            command = f"cat {file_path}"
            response = subprocess.check_output(command, shell=True, text=True)
            return response[:20000]
        except Exception as e:
            return f"Error in reading file: {file_path}, Error: {str(e)}"
        
class ExecuteCommandTool(AbstractTool):
    name: str = "execute_command"
    description: str = """
Execute a command in the system shell. Use this function when there is a need to run a system command, and execute programs.
Args:
    command: The command to execute in the system shell.
Returns:
    A string representation of the exit code and output of the command.
"""
    parameters: dict[str, dict[str, str]] = {
        "command": {
            "type": "string",
            "description": "The command to execute in the system shell."
        }
    }

    def __call__(self, command: str, *args, **kwargs):    
        try:
            # always execute commands in the AI_RESEARCHER_AGENT_WORKING_PATH path
            command = f"cd {Config.AI_RESEARCHER_AGENT_WORKING_PATH} && {command}"
            # get string output of the command
            response = subprocess.check_output(command, shell=True, text=True)
            return response
        except Exception as e:
            return f"Error running command: {str(e)}"

if __name__ == "__main__":
    code_structure_tool = CodeStructureTool()
    print(code_structure_tool(directory=Config.WORKSPACE_PATH))
    
    read_file_tool = ReadFileTool()
    print(read_file_tool(file_path=os.path.join(Config.WORKSPACE_PATH, "src/ai_researcher/classes/AbstractTool.py")))

    execute_command_tool = ExecuteCommandTool()
    print(execute_command_tool(command=f"ls -la {Config.WORKSPACE_PATH}"))
