import json, platform, re

# 'config_manager' and 'chat_gpt_api' (if needed) are injected into the global scope.
# No need to 'import config'

# ChatGPT-GUI plugin: Instruct ChatGPT to excute python code directly in response to user input
# Written by: Eliran Wong
# Feature: Non-python users can use natural language to instruct ChatGPT to perform whatever tasks which python is capable to do.

# Usage:
# 1. Select "Execute Python Code" as predefined context
# 2. Use natural language to instruct ChatGPT to execute what python is capable to do
# 
# Examples, try:
# Tell me the current time.
# Tell me how many files in the current directory.
# What is my operating system and version?
# Is google chrome installed on this computer?
# Open web browser.
# Open https://github.com/eliranwong/ChatGPT-GUI in a web browser.
# Search ChatGPT in a web browser.
# Open the current directory using the default file manager.
# Open VLC player.
# Open Calendar

def run_python(function_args):

    def fineTunePythonCode(code):
        # config_manager is available in the global scope of the plugin execution
        insert_string = "config_manager.update_setting('python_function_response', " # Python code string
        code = re.sub("^!(.*?)$", r"import os\nos.system(\1)", code, flags=re.M)
        if "\n" in code:
            substrings = code.rsplit("\n", 1)
            lastLine = re.sub(r"print\((.*)\)", r"\1", substrings[-1])
            # Ensure the value part of update_setting is closed with a parenthesis
            code = code if lastLine.startswith(" ") else f"{substrings[0]}\n{insert_string}{lastLine})"
        else:
            code = f"{insert_string}{code})" # Close parenthesis
        return code

    # retrieve argument values from a dictionary
    #print(function_args)
    function_args_code = function_args.get("code") # required
    new_function_args = fineTunePythonCode(function_args_code)

    # The exec call will use the plugin's global scope, which has config_manager
    # Ensure python_function_response is reset or handled if exec fails before update_setting
    config_manager.update_setting('python_function_response', '') # Clear previous response
    try:
        exec(new_function_args, globals()) # globals() here refers to plugin's globals
        function_response_val = str(config_manager.get_setting('python_function_response'))
    except Exception as e:
        print(f"Error executing dynamic python code: {e}")
        function_response_val = function_args_code # Fallback to original code on error
    info = {"information": function_response_val}
    function_response = json.dumps(info)
    return json.dumps(info)

functionSignature = {
    "name": "run_python",
    "description": "Execute python code",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "python code, e.g. print('Hello world')",
            },
        },
        "required": ["code"],
    },
}

# Update chat_gpt_api_function_signatures
signatures = config_manager.get_setting('chat_gpt_api_function_signatures')
if not isinstance(signatures, list):
    signatures = []
signatures.append(functionSignature)
config_manager.update_setting('chat_gpt_api_function_signatures', signatures)

# Update chat_gpt_api_available_functions
available_functions = config_manager.get_setting('chat_gpt_api_available_functions')
if not isinstance(available_functions, dict):
    available_functions = {}
available_functions["run_python"] = run_python
config_manager.update_setting('chat_gpt_api_available_functions', available_functions)

current_platform = platform.system()
if current_platform == "Darwin":
    current_platform = "macOS"

# Update predefined_contexts
contexts = config_manager.get_setting('predefined_contexts')
if not isinstance(contexts, dict):
    contexts = {}
contexts["Execute Python Code"] = f"""I running {current_platform} on this device. Execute python codes directly on my behalf to achieve the following tasks. Do not show me the codes unless I explicitly request."""
config_manager.update_setting('predefined_contexts', contexts)
