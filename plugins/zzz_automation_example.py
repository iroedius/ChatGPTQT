import re # openai import removed, config import removed
from datetime import datetime

# 'config_manager' and 'chat_gpt_api' are injected into the global scope.

"""
A plugin example to generate multiple chat responses and save in database file.
This plugin will now use the active LLM provider via chat_gpt_api.llm_provider.
Note: The original plugin's support for n > 1 choices is not directly supported
by the current llm_provider.generate_response(), so it will generate one response.
"""

def loadResponses(predefinedContextValue, userInput): # Renamed predefinedContext to avoid conflict
    # title
    title = re.sub("\n", " ", userInput)[:50]

    # Prepare chat history and prompt for llm_provider.generate_response
    # The llm_provider's implementation will handle system prompts and its own context logic.
    # This plugin can optionally pass a simplified history or context string if needed.

    # Simplified history preparation for this plugin's purpose:
    # The predefinedContextValue could be used as an initial user message for context.
    chat_history_for_provider = []
    if predefinedContextValue: # Assuming predefinedContextValue is a string from the contexts dict
        chat_history_for_provider.append({'role': 'user', 'content': predefinedContextValue})
        # Optionally, add an empty assistant response to prime conversation for some models
        # chat_history_for_provider.append({'role': 'assistant', 'content': ''})

    # The userInput is the current prompt
    current_prompt = userInput

    # Get temperature and max_tokens from config for the current provider
    # These might be specific to openai or gemini, so care should be taken
    # or providers should handle None and use their internal defaults.
    temperature = config_manager.get_setting('chat_gpt_api_temperature') # Example, might need provider-specific key
    max_tokens = config_manager.get_setting('chat_gpt_api_max_tokens')   # Example

    raw_responses_text = f">>> {userInput}\n\n" # Start with the prompt

    try:
        # Use the injected chat_gpt_api instance to access the llm_provider
        if not chat_gpt_api or not hasattr(chat_gpt_api, 'llm_provider'):
            raise ValueError("LLM provider not available via chat_gpt_api instance.")

        generated_text = chat_gpt_api.llm_provider.generate_response(
            prompt=current_prompt,
            chat_history=chat_history_for_provider, # Pass the constructed history
            temperature=temperature,
            max_tokens=max_tokens
        )

        if generated_text.startswith("Error:"): # Provider returned an error string
             raw_responses_text += f"LLM Error: {generated_text}\n\n"
        else:
             raw_responses_text += f"{generated_text}\n\n"

    except Exception as e:
        # Handle other exceptions during the call
        print(f"Plugin Error during LLM call: {e}")
        raw_responses_text += f"Plugin Error: {e}\n\n"

    # Process responses (assuming chat_gpt_transformers is a list of functions)
    # This part might need review if transformers expect specific OpenAI response structures
    processed_responses = raw_responses_text
    transformers = config_manager.get_setting('chat_gpt_transformers')
    if isinstance(transformers, list):
        for t in transformers:
            try:
                processed_responses = t(processed_responses)
            except Exception as e:
                print(f"Error applying transformer {t}: {e}")

    processed_responses = re.sub("\n\n[\n]+?([^\n])", r"\n\n\1", processed_responses)

    id_val = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Use the injected chat_gpt_api instance to access database
    if chat_gpt_api and hasattr(chat_gpt_api, 'database'):
        chat_gpt_api.database.insert(id_val, title, processed_responses)
    else:
        print("Error: chat_gpt_api instance or database not available in plugin.")


# automate saving responses in database file
userInputs = (
    #"Tell me about ChatGPT",
)
predefinedContexts = (
    "[none]",
)
if userInputs:
    for userInput in userInputs:
        for predefinedContext in predefinedContexts:
            try:
                loadResponses(predefinedContext, userInput)
            except:
                print(f"Failed processing '{predefinedContext}' - '{userInput}'!")
