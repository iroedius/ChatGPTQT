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
            ('chatGPTApiAudio', 0),
            ('chatGPTApiAudioLanguage', 'en'),
            ('chatGPTApiModel', 'gpt-4-turbo-preview'),
            ('chatGPTApiPredefinedContext', '[none]'),
            ('chatGPTApiContext', ''),
            ('chatGPTApiLastChatDatabase', ''),
            ('chatGPTApiMaxTokens', 4096),
            ('chatGPTApiNoOfChoices', 1),
            ('chatGPTApiTemperature', 0.8),
            ('chatGPTApiFunctionCall', 'none'),
            ('chatAfterFunctionCalled', True),
            ('runPythonScriptGlobally', False),
            ('darkTheme', True),
            ('developer', False),
            ('enableSystemTray', True),
            ('fontSize', 14),
            ('pocketsphinxModelPath', ''),
            ('pocketsphinxModelPathBin', ''),
            ('pocketsphinxModelPathDict', ''),
            ('regexpSearchEnabled', True),
            # 'includeDuckDuckGoSearchResults', False,
            # 'maximumDuckDuckGoSearchResults', 5,
            ('loadingInternetSearches', 'none'),
            ('maximumInternetSearchResults', 5),
            ('chatGPTApiContextInAllInputs', False),
            ('chatGPTApiAutoScrolling', True),
            ('chatGPTPluginExcludeList', ['testing_function_calling', 'zzz_automation_example']),
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

    def get_internal(self, key: str) -> Any:
        return self.internal.get(key)

    def update_internal(self, key: str, value: Any) -> None:
        self.internal[key] = value

    def save_config(self) -> None:
        with Path(self.settings_path).open('w') as f:
            json.dump(self.settings, f, indent=2)
