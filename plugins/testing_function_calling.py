import json

# 'config_manager' and 'chat_gpt_api' (if needed) are injected into the global scope.
# No need to 'import config'

# modified from source: https://platform.openai.com/docs/guides/gpt/function-calling

def get_current_weather(function_args):
    # retrieve argument values from a dictionary
    location = function_args.get("location") # required
    unit=function_args.get("unit", "fahrenheit") # optional

    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

functionSignature = {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
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
available_functions["get_current_weather"] = get_current_weather
config_manager.update_setting('chat_gpt_api_available_functions', available_functions)
