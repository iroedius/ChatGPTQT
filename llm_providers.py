import abc
import openai
import os # For environment variables if needed for API keys
import google.generativeai as genai
from config_manager import ConfigManager

class LLMProvider(abc.ABC):
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.api_key = None
        self.model_name = None
        # Other common config might be initialized here or in subclasses

    @abc.abstractmethod
    def load_config(self) -> None:
        """Loads provider-specific configuration from the config_manager."""
        pass

    @abc.abstractmethod
    def get_available_models(self) -> list[str]:
        """Returns a list of available model names for this provider."""
        pass

    @abc.abstractmethod
    def generate_response(self, prompt: str, chat_history: list[dict] | None = None, temperature: float | None = None, max_tokens: int | None = None) -> str:
        """
        Generates a response from the LLM.
        chat_history is a list of dictionaries, e.g., [{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello!'}]
        Returns the text content of the response.
        """
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        self.organization = None
        self.temperature = 0.8 # Default, will be overridden by load_config
        self.max_tokens = 4096 # Default, will be overridden by load_config
        self.load_config() # Load config during initialization

    def load_config(self) -> None:
        self.api_key = self.config_manager.get_setting('openai_api_key')
        self.organization = self.config_manager.get_setting('openai_api_organization')
        self.model_name = self.config_manager.get_setting('chat_gpt_api_model') # Assuming this is the OpenAI model
        self.temperature = self.config_manager.get_setting('chat_gpt_api_temperature')
        self.max_tokens = self.config_manager.get_setting('chat_gpt_api_max_tokens')

        if self.api_key:
            openai.api_key = self.api_key
            if self.organization:
                openai.organization = self.organization
        else:
            # Potentially raise an error or handle missing API key
            print("Warning: OpenAI API Key is not configured.")


    def get_available_models(self) -> list[str]:
        # For now, a static list. Could be fetched from API in the future.
        return [
            'gpt-4-turbo-preview',
            'gpt-4-0125-preview',
            'gpt-4-1106-preview',
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-0125',
            'gpt-3.5-turbo-1106'
        ]

    def _get_openai_context_string(self) -> str:
        # Helper to get context string, similar to old ChatGPTAPI.get_context
        # This is OpenAI specific regarding key names.
        context_str = ""
        predefined_contexts_setting = self.config_manager.get_setting('predefined_contexts')
        if isinstance(predefined_contexts_setting, str): # Should be dict, but guard
            try:
                predefined_contexts_setting = eval(predefined_contexts_setting) # Use eval if it's a string representation of dict
            except:
                predefined_contexts_setting = {}

        current_predefined_key = self.config_manager.get_setting('chat_gpt_api_predefined_context')

        if current_predefined_key == '[custom]':
            context_str = self.config_manager.get_setting('chat_gpt_api_context', '')
        elif current_predefined_key != '[none]' and isinstance(predefined_contexts_setting, dict):
            context_str = predefined_contexts_setting.get(current_predefined_key, '')

        # Specific handling for 'Execute Python Code' context affecting function calls (OpenAI specific)
        # This might be better placed directly where 'functions' and 'function_call' params are set up for API call
        # if current_predefined_key == 'Execute Python Code':
        #     if self.config_manager.get_setting('chat_gpt_api_function_call') == 'none':
        #         self.config_manager.update_setting('chat_gpt_api_function_call', 'auto') # Careful: modifying config here
        #     if self.config_manager.get_setting('loading_internet_searches') == 'always':
        #         self.config_manager.update_setting('loading_internet_searches', 'auto') # Careful: modifying config here
        return context_str

    def _prepare_messages(self, prompt: str, chat_history: list[dict] | None = None) -> list[dict]:
        system_message_content = "You're a kind helpful assistant."
        # OpenAI specific function call hints in system message - consider if this should be more dynamic
        # For now, assuming function call config is read directly by generate_response if needed.
        # if self.config_manager.get_setting('chat_gpt_api_function_call') == 'auto' and \
        #    self.config_manager.get_setting('chat_gpt_api_function_signatures'):
        #     system_message_content += ' Only use the functions you have been provided with.'

        messages = [{'role': 'system', 'content': system_message_content}]

        context_string = self._get_openai_context_string()

        # How context is injected depends on chat_gpt_api_context_in_all_inputs
        context_in_all_inputs = self.config_manager.get_setting('chat_gpt_api_context_in_all_inputs', False)

        if context_string:
            if not chat_history and not context_in_all_inputs: # Context as first assistant message if no history and not "all inputs"
                 messages.append({'role': 'assistant', 'content': context_string})
            # If context_in_all_inputs, it will be prepended to the user's prompt later.

        if chat_history:
            messages.extend(chat_history)

        current_prompt = prompt
        if context_string and context_in_all_inputs:
            current_prompt = f"{context_string}\n{prompt}"
        elif context_string and not chat_history and context_in_all_inputs : # if no history but context_in_all_inputs is true
             current_prompt = f"{context_string}\n{prompt}"


        messages.append({'role': 'user', 'content': current_prompt})
        return messages

    def generate_response(self, prompt: str, chat_history: list[dict] | None = None, temperature: float | None = None, max_tokens: int | None = None) -> str:
        if not self.api_key:
            return "Error: OpenAI API Key is not configured."

        messages = self._prepare_messages(prompt, chat_history)

        current_temp = temperature if temperature is not None else self.temperature
        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        try:
            completion = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=current_temp,
                max_tokens=current_max_tokens,
                # n=1, # Default is 1, can be made configurable if needed
                # stream=False # For now, non-streaming. Streaming would change the return type.
            )
            response_content = completion.choices[0].message.content
            return response_content.strip() if response_content else ""
        except openai.error.APIError as e:
            return f"OpenAI API Error: {e}"
        except openai.error.APIConnectionError as e:
            return f"OpenAI Connection Error: {e}"
        except openai.error.RateLimitError as e:
            return f"OpenAI Rate Limit Error: {e}"
        except Exception as e:
            return f"An unexpected error occurred with OpenAI: {e}"

class GeminiProvider(LLMProvider):
    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager)
        # self.model_name will be set in load_config, api_key in base class
        self.load_config()

    def load_config(self) -> None:
        self.api_key = self.config_manager.get_setting('gemini_api_key')
        self.model_name = self.config_manager.get_setting('gemini_model_name')

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
            except Exception as e:
                print(f"Error configuring Gemini API: {e}")
                self.api_key = None # Mark as not configured if error
        else:
            print("Warning: Gemini API Key is not configured.")

    def _get_gemini_context_string(self) -> str:
        # Using same config keys as OpenAI for context for simplicity.
        # This can be changed to gemini_specific_context keys if needed.
        context_str = ""
        predefined_contexts_setting = self.config_manager.get_setting('predefined_contexts')
        if isinstance(predefined_contexts_setting, str): # Should be dict, but guard
            try:
                predefined_contexts_setting = eval(predefined_contexts_setting)
            except:
                predefined_contexts_setting = {}

        current_predefined_key = self.config_manager.get_setting('chat_gpt_api_predefined_context')

        if current_predefined_key == '[custom]':
            context_str = self.config_manager.get_setting('chat_gpt_api_context', '')
        elif current_predefined_key != '[none]' and isinstance(predefined_contexts_setting, dict):
            context_str = predefined_contexts_setting.get(current_predefined_key, '')
        return context_str

    def get_available_models(self) -> list[str]:
        # Static list for now, can be expanded or fetched via API if available later
        return ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-pro-latest", "gemini-pro-vision"]

    def _prepare_chat_history(self, chat_history: list[dict] | None = None, initial_context_str: str = "") -> list[dict]:
        transformed_history = []

        # Prepend initial context as a user message, then an empty model response to prime the conversation
        if initial_context_str:
            transformed_history.append({'role': 'user', 'parts': [{'text': initial_context_str}]})
            # Add an empty model response to ensure the next message is 'user' if chat_history is empty
            # However, Gemini's start_chat(history=...) can handle starting with 'user' if history is properly formed.
            # If chat_history is empty, the first actual user prompt will follow this.
            # If chat_history is NOT empty, this context is prepended.
            # Let's test if an empty model part is needed or if Gemini handles it.
            # For safety, adding an empty model part if history is completely empty after context.
            if not chat_history:
                 transformed_history.append({'role': 'model', 'parts': [{'text': ''}]})


        if chat_history:
            for message in chat_history:
                role = message.get('role')
                content = message.get('content')
                if role and content:
                    # Map 'assistant' to 'model' for Gemini
                    gemini_role = 'model' if role == 'assistant' else role
                    transformed_history.append({
                        'role': gemini_role,
                        'parts': [{'text': content}]
                    })
        return transformed_history

    def generate_response(self, prompt: str, chat_history: list[dict] | None = None, temperature: float | None = None, max_tokens: int | None = None) -> str:
        if not self.api_key: # This is the gemini_api_key loaded in self.api_key
            return "Error: Gemini API Key is not configured."
        if not self.model_name:
            return "Error: Gemini model name is not configured."

        try:
            context_string = self._get_gemini_context_string()
            context_in_all_inputs = self.config_manager.get_setting('chat_gpt_api_context_in_all_inputs', False)

            current_prompt = prompt
            # Prepend context to current prompt if 'context_in_all_inputs' is true
            if context_string and context_in_all_inputs:
                current_prompt = f"{context_string}\n{prompt}"

            # Prepare history: if context is not for all inputs, it's an initial priming message.
            initial_context_for_history = ""
            if context_string and not context_in_all_inputs:
                initial_context_for_history = context_string

            transformed_history = self._prepare_chat_history(chat_history, initial_context_str=initial_context_for_history)

            # Configuration for generation - Gemini uses GenerationConfig
            generation_config_params = {}
            if temperature is not None:
                generation_config_params['temperature'] = temperature
            # Gemini uses 'max_output_tokens' within GenerationConfig
            if max_tokens is not None:
                generation_config_params['max_output_tokens'] = max_tokens

            generation_config = genai.types.GenerationConfig(**generation_config_params) if generation_config_params else None

            model_instance = genai.GenerativeModel(self.model_name)

            # Start chat or generate content
            # The 'current_prompt' now includes context if 'context_in_all_inputs' was true.
            # 'transformed_history' includes context if 'context_in_all_inputs' was false.
            if not transformed_history: # Or if preferring generate_content for single turns always
                 full_contents_for_generate = [{'role': 'user', 'parts': [{'text': current_prompt}]}]
                 # If there was an initial_context_for_history and no other chat_history,
                 # transformed_history would not be empty. So this case is for when there's truly no preceding context/history.
                 if initial_context_for_history : # Should not happen if logic is correct, but as a safeguard
                      full_contents_for_generate = [{'role': 'user', 'parts': [{'text': initial_context_for_history}]}, {'role': 'model', 'parts': [{'text': ''}]}, {'role': 'user', 'parts': [{'text': current_prompt}]}]

                 gemini_response = model_instance.generate_content(
                     contents=full_contents_for_generate,
                     generation_config=generation_config
                 )
            else:
                chat_session = model_instance.start_chat(history=transformed_history)
                gemini_response = chat_session.send_message(
                    content=[{'text': current_prompt}], # current_prompt already has context if context_in_all_inputs
                    generation_config=generation_config
                )

            # Check for empty candidates or parts before accessing .text
            if gemini_response.candidates and gemini_response.candidates[0].content.parts:
                return gemini_response.text
            else:
                # Investigate response structure if text is not directly available
                # print(f"Gemini raw response: {gemini_response}") # For debugging
                if gemini_response.prompt_feedback and gemini_response.prompt_feedback.block_reason:
                    return f"Error: Gemini content generation blocked. Reason: {gemini_response.prompt_feedback.block_reason_message or gemini_response.prompt_feedback.block_reason}"
                return "Error: Gemini response was empty or malformed."

        except Exception as e:
            # Attempt to get more detailed error message if available
            # error_details = getattr(e, 'message', str(e)) # Basic
            # if hasattr(e, 'response') and hasattr(e.response, 'text'): # For potential HTTP errors
            #     error_details = f"{error_details} - {e.response.text}"
            return f"Gemini API Error: {e}"

if __name__ == '__main__':
    print("llm_providers.py loaded. Contains LLMProvider, OpenAIProvider, and GeminiProvider.")

    # Attempt to use the real ConfigManager, assuming settings.json might exist
    # This test will be more meaningful if API keys are in settings.json
    try:
        config_manager_real = ConfigManager() # Assumes settings.json is in the same dir or default path
        print("Successfully instantiated ConfigManager.")
    except FileNotFoundError:
        print("settings.json not found. Cannot perform live tests with ConfigManager.")
        config_manager_real = None # Ensure it's defined for later checks
    except Exception as e:
        print(f"Error instantiating ConfigManager: {e}")
        config_manager_real = None

    if config_manager_real:
        # Test OpenAIProvider
        print("\n--- Testing OpenAIProvider with ConfigManager ---")
        try:
            openai_provider = OpenAIProvider(config_manager_real)
            print(f"OpenAIProvider instantiated. API Key configured: {bool(openai_provider.api_key)}")
            print(f"OpenAI Model: {openai_provider.model_name}")
            print(f"Available OpenAI Models: {openai_provider.get_available_models()}")

            if openai_provider.api_key:
                test_prompt_openai = "Hello, OpenAI! Write a haiku about APIs."
                print(f"\nTesting OpenAI generate_response with prompt: '{test_prompt_openai}'")
                response_openai = openai_provider.generate_response(test_prompt_openai)
                print(f"OpenAI Response: {response_openai}")
            else:
                print("OpenAI API key not found in settings.json or by ConfigManager. Skipping generate_response test.")
        except Exception as e:
            print(f"Error during OpenAIProvider test: {e}")

        # Test GeminiProvider
        print("\n--- Testing GeminiProvider with ConfigManager ---")
        try:
            gemini_provider = GeminiProvider(config_manager_real)
            print(f"GeminiProvider instantiated. API Key configured: {bool(gemini_provider.api_key)}")
            print(f"Gemini Model: {gemini_provider.model_name}")
            print(f"Available Gemini Models: {gemini_provider.get_available_models()}")

            if gemini_provider.api_key:
                test_prompt_gemini = "Hello, Gemini! Write a haiku about LLMs."
                print(f"\nTesting Gemini generate_response with prompt: '{test_prompt_gemini}'")
                response_gemini = gemini_provider.generate_response(test_prompt_gemini, temperature=0.7, max_tokens=100)
                print(f"Gemini Response: {response_gemini}")
            else:
                print("Gemini API key not found in settings.json or by ConfigManager. Skipping generate_response test.")
        except Exception as e:
            print(f"Error during GeminiProvider test: {e}")
    else:
        print("Skipping provider tests with real ConfigManager as it could not be initialized.")

    # Fallback to Mock tests if real ConfigManager failed or for local env var testing
    # This part is mostly for local dev, might not be as useful in sandbox if keys aren't in env
    print("\n--- Fallback/Mock Tests (using environment variables if available) ---")
    class MockConfigManagerForEnv:
        def get_setting(self, key, default=None):
            if key == 'openai_api_key': return os.environ.get("OPENAI_API_KEY")
            if key == 'chat_gpt_api_model': return 'gpt-3.5-turbo'
            if key == 'chat_gpt_api_temperature': return 0.7
            if key == 'chat_gpt_api_max_tokens': return 150
            if key == 'gemini_api_key': return os.environ.get("GEMINI_API_KEY")
            if key == 'gemini_model_name': return 'gemini-pro'
            # Required for context handling in providers
            if key == 'predefined_contexts': return {"[none]": "", "[custom]": ""}
            if key == 'chat_gpt_api_predefined_context': return "[none]"
            if key == 'chat_gpt_api_context': return ""
            if key == 'chat_gpt_api_context_in_all_inputs': return False
            return default

    mock_config_env = MockConfigManagerForEnv()

    # Test OpenAIProvider with Env Vars
    if mock_config_env.get_setting('openai_api_key'):
        print("\n--- Testing OpenAIProvider with Environment Variable ---")
        openai_provider_env = OpenAIProvider(mock_config_env)
        if openai_provider_env.api_key:
            response_openai_env = openai_provider_env.generate_response("Test OpenAI from env.")
            print(f"OpenAI Env Response: {response_openai_env}")
        else:
            print("OpenAI API key from env not loaded by provider.")
    else:
        print("\nOPENAI_API_KEY environment variable not set. Skipping OpenAI env test.")

    # Test GeminiProvider with Env Vars
    if mock_config_env.get_setting('gemini_api_key'):
        print("\n--- Testing GeminiProvider with Environment Variable ---")
        gemini_provider_env = GeminiProvider(mock_config_env)
        if gemini_provider_env.api_key:
            response_gemini_env = gemini_provider_env.generate_response("Test Gemini from env.")
            print(f"Gemini Env Response: {response_gemini_env}")
        else:
            print("Gemini API key from env not loaded by provider.")
    else:
        print("\nGEMINI_API_KEY environment variable not set. Skipping Gemini env test.")
