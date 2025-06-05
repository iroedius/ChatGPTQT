import json, googlesearch

# 'config_manager' and 'chat_gpt_api' (if needed) are injected into the global scope.
# No need to 'import config'

# Use google https://pypi.org/project/googlesearch-python/ to search internet for information, about which ChatGPT doesn't know.

def integrate_google_searches(function_args):
    # retrieve argument values from a dictionary
    #print(function_args)
    keywords = function_args.get("keywords") # required

    info = {}
    # Use config_manager to get the setting
    max_results = config_manager.get_setting('maximum_internet_search_results', 5) # Default to 5 if not set
    for index, item in enumerate(googlesearch.search(keywords, advanced=True, num_results=max_results)):
        info[f"information {index}"] = {
            "title": item.title,
            "url": item.url,
            "description": item.description,
        }
    return json.dumps(info)

functionSignature = {
    "name": "integrate_google_searches",
    "description": "Search internet for keywords when ChatGPT does not have information",
    "parameters": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "keywords for searches, e.g. ChatGPT",
            },
        },
        "required": ["keywords"],
    },
}

config_manager.update_setting('integrate_google_searches_signature', [functionSignature])

# Update chat_gpt_api_function_signatures
signatures = config_manager.get_setting('chat_gpt_api_function_signatures')
if not isinstance(signatures, list):
    signatures = []
# Ensure it's not added multiple times if plugin re-runs, though append is usually fine for function defs
# A more robust way would be to check if a signature with this name already exists.
signatures.insert(0, functionSignature)
config_manager.update_setting('chat_gpt_api_function_signatures', signatures)

# Update chat_gpt_api_available_functions
available_functions = config_manager.get_setting('chat_gpt_api_available_functions')
if not isinstance(available_functions, dict):
    available_functions = {}
available_functions["integrate_google_searches"] = integrate_google_searches
config_manager.update_setting('chat_gpt_api_available_functions', available_functions)
