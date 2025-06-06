import json
from pathlib import Path
from typing import Any


class ConfigManager:
    def __init__(self, settings_path: str = 'settings.json', translations_path: str = 'translations.json') -> None:
        self.settings_path = Path(settings_path)
        self.settings = self.load_settings()
        self.translations_path = Path(translations_path)
        self.translations = self.load_translations()
        self.internal = {}

        default_settings = (
            ('chat_gpt_api_audio', 0),
            ('chat_gpt_api_audio_language', 'en'),
            ('chat_gpt_api_model', 'gpt-4-turbo-preview'),
            ('chat_gpt_api_predefined_context', '[none]'),
            ('chat_gpt_api_context', ''),
            ('chat_gpt_api_last_chat_database', ''),
            ('chat_gpt_api_max_tokens', 4096),
            ('chat_gpt_api_no_of_choices', 1),
            ('chat_gpt_api_temperature', 0.8),
            ('chat_gpt_api_function_call', 'none'),
            ('chat_after_function_called', True),
            ('run_python_script_globally', False),
            ('dark_theme', True),
            ('developer', False),
            ('enable_system_tray', True),
            ('font_size', 14),
            ('pocketsphinx_model_path', ''),
            ('pocketsphinx_model_path_bin', ''),
            ('pocketsphinx_model_path_dict', ''),
            ('regexp_search_enabled', True),
            # 'includeDuckDuckGoSearchResults', False,
            # 'maximumDuckDuckGoSearchResults', 5,
            ('loading_internet_searches', 'none'),
            ('maximum_internet_search_results', 5),
            ('chat_gpt_api_context_in_all_inputs', False),
            ('chat_gpt_api_auto_scrolling', True),
            ('chat_gpt_plugin_exclude_list', ['testing_function_calling', 'zzz_automation_example']),
            ('predefined_contexts', { "[none]": "", "[custom]": "" }),
            ('input_suggestions', []),
            ('chat_gpt_transformers', []),
            ('chat_gpt_api_function_signatures', []),
            ('chat_gpt_api_available_functions', {}),
            ('python_function_response', ''),
            ('integrate_google_searches_signature', []),
            ('gemini_api_key', ''),
            ('gemini_model_name', 'gemini-pro'),
            ('selected_llm_provider', 'openai'),
        )

        for key, value in default_settings:
            if key not in self.settings:
                # newvalue = pprint.pformat(value)
                print(f'Changing setting {key} with value {self.settings.get(key)} to the default value {value}')
                self.settings[key] = value

    def load_settings(self) -> dict:
        with self.settings_path.open('r') as f:
            return json.load(f)

    def load_translations(self) -> dict:
        with self.translations_path.open('r') as f:
            return json.load(f)

    def get_translation(self, key: str, default: str | None = None) -> str:
        return self.translations.get(key, default)

    def get_setting(self, key: str, default: str | None = None) -> Any:
        return self.settings.get(key, default)

    def update_setting(self, key: str, value: str | float | bool | list) -> None:
        self.settings[key] = value
        self.save_config()

    def save_config(self) -> None:
        with Path(self.settings_path).open('w') as f:
            json.dump(self.settings, f, indent=2)
