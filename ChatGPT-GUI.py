#!/usr/bin/env python

# import ctypes
from __future__ import annotations

# try:
#     from pocketsphinx import LiveSpeech, get_model_path
#     sphinx_installed = True
# except ImportWarning:
#     sphinx_installed = False
import contextlib
import ctypes
import glob
import locale
import logging
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import traceback
import urllib.parse
import webbrowser
from ast import literal_eval
from datetime import datetime
from functools import partial
from io import StringIO
from pathlib import Path
from shutil import copyfile
from typing import Any

import openai
import pyqtdarktheme as qdarktheme
import tiktoken
from gtts import gTTS
from PySide6.QtCore import QModelIndex, QRegularExpression, Qt, QThread, Signal
from PySide6.QtGui import QAction, QFontMetrics, QGuiApplication, QIcon, QStandardItem, QStandardItemModel, QTextDocument
from PySide6.QtPrintSupport import QPrintDialog, QPrinter
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from config_manager import ConfigManager
from util.Worker import ChatGPTResponse, OpenAIImage
from llm_providers import LLMProvider, OpenAIProvider, GeminiProvider # Added GeminiProvider

config_manager = ConfigManager()

this_file = Path(__file__).resolve()
wd = this_file.parent
if Path.cwd() != wd:
    os.chdir(wd)
if not Path('config.py').is_file():
    Path('config.py').open('a').close()


class SpeechRecognitionThread(QThread):
    phrase_recognized = Signal(str)

    def __init__(self, parent: QThread) -> None:
        super().__init__(parent)
        self.is_running = False

    def run(self) -> None:
        self.is_running = True
        # if config_manager.get_setting('pocketsphinx_model_path'):
        #     # download English dictionary at: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
        #     # download voice models at https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/
        #     speech = LiveSpeech(
        #         # sampling_rate=16000,  # optional
        #         hmm=get_model_path(config_manager.get_setting('pocketsphinx_model_path')),
        #         lm=get_model_path(config_manager.get_setting('pocketsphinx_model_path_bin')),
        #         dic=get_model_path(config_manager.get_setting('pocketsphinx_model_path_dict')),
        #     )
        # else:
        #     speech = LiveSpeech()

        # for phrase in speech:
        #     if not self.is_running:
        #         break
        #     recognized_text = str(phrase)
        #     self.phrase_recognized.emit(recognized_text)

    def stop(self) -> None:
        self.is_running = False


class ApiDialog(QDialog):
    def __init__(self, parent: None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(config_manager.get_translation('settings'))

        self.apiKeyEdit = QLineEdit(config_manager.get_setting('openaiApiKey'))
        self.apiKeyEdit.setEchoMode(QLineEdit.Password)
        self.orgEdit = QLineEdit(config_manager.get_setting('openaiApiOrganization'))
        self.orgEdit.setEchoMode(QLineEdit.Password)
        self.apiModelBox = QComboBox()
        initial_index = 0
        for count, value in enumerate({'gpt-4-turbo-preview', 'gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106'}):
            self.apiModelBox.addItem(value)
            if value == config_manager.get_setting('chatGPTApiModel'):
                initial_index = count
        self.apiModelBox.setCurrentIndex(initial_index)
        self.functionCallingBox = QComboBox()
        initial_index = 0
        for count, value in enumerate({'auto', 'none'}):
            self.functionCallingBox.addItem(value)
            if value == config_manager.get_setting('chatGPTApiFunctionCall'):
                initial_index = count
        self.functionCallingBox.setCurrentIndex(initial_index)
        self.loadingInternetSearchesBox = QComboBox()
        initial_index = 0
        for count, value in enumerate({'always', 'auto', 'none'}):
            self.loadingInternetSearchesBox.addItem(value)
            if value == config_manager.get_setting('loadingInternetSearches'):
                initial_index = count
        self.loadingInternetSearchesBox.setCurrentIndex(initial_index)
        self.maxTokenEdit = QLineEdit(str(config_manager.get_setting('chatGPTApiMaxTokens')))
        self.maxTokenEdit.setToolTip('''The maximum number of tokens to generate in the completion.
The token count of your prompt plus max_tokens cannot exceed the model's context length.
Most models have a context length of 2048 tokens (except for the newest models, which support 4096).''')
        self.maxInternetSearchResults = QLineEdit(str(config_manager.get_setting('maximumInternetSearchResults')))
        self.maxInternetSearchResults.setToolTip('The maximum number of internet search response to be included.')
        # self.includeInternetSearches = QCheckBox(config_manager.get_setting("this_translation").include)
        # self.includeInternetSearches.setToolTip("Include latest internet search results")
        # self.includeInternetSearches.setCheckState(Qt.Checked if config_manager.get_setting("includeDuckDuckGoSearchResults") else Qt.Unchecked)
        # self.includeDuckDuckGoSearchResults = config_manager.get_setting("includeDuckDuckGoSearchResults")
        self.autoScrollingCheckBox = QCheckBox(config_manager.get_translation('enable'))
        self.autoScrollingCheckBox.setToolTip('Auto-scroll display as responses are received')
        self.autoScrollingCheckBox.setCheckState(Qt.Checked if config_manager.get_setting('chatGPTApiAutoScrolling') else Qt.Unchecked)
        self.chatGPTApiAutoScrolling = config_manager.get_setting('chatGPTApiAutoScrolling')
        self.chatAfterFunctionCalledCheckBox = QCheckBox(config_manager.get_translation('enable'))
        self.chatAfterFunctionCalledCheckBox.setToolTip('Automatically generate next chat response after a function is called')
        self.chatAfterFunctionCalledCheckBox.setCheckState(Qt.Checked if config_manager.get_setting('chatAfterFunctionCalled') else Qt.Unchecked)
        self.chatAfterFunctionCalled = config_manager.get_setting('chatAfterFunctionCalled')
        self.runPythonScriptGloballyCheckBox = QCheckBox(config_manager.get_translation('enable'))
        self.runPythonScriptGloballyCheckBox.setToolTip('Run user python script in global scope')
        self.runPythonScriptGloballyCheckBox.setCheckState(Qt.Checked if config_manager.get_setting('runPythonScriptGlobally') else Qt.Unchecked)
        self.runPythonScriptGlobally = config_manager.get_setting('runPythonScriptGlobally')
        self.contextEdit = QLineEdit(config_manager.get_setting('chatGPTApiContext'))
        first_input_only = config_manager.get_translation('firstInputOnly')
        all_inputs = config_manager.get_translation('allInputs')
        self.applyContextIn = QComboBox()
        self.applyContextIn.addItems([first_input_only, all_inputs])
        self.applyContextIn.setCurrentIndex(1 if config_manager.get_setting('chatGPTApiContextInAllInputs') else 0)
        self.predefinedContextBox = QComboBox()
        initial_index = 0
        index = 0
        # TODO: Ensure 'predefined_contexts' is a dictionary as expected.
        # If it's a string representation of a dict, it needs parsing first.
        # For now, assuming it's already a dictionary.
        predefined_contexts_setting = config_manager.get_setting('predefined_contexts')
        if isinstance(predefined_contexts_setting, str):
            try:
                predefined_contexts_setting = literal_eval(predefined_contexts_setting)
            except (ValueError, SyntaxError):
                predefined_contexts_setting = {} # Default to empty if parsing fails

        for key, value in predefined_contexts_setting.items():
            self.predefinedContextBox.addItem(key)
            self.predefinedContextBox.setItemData(self.predefinedContextBox.count() - 1, value, role=Qt.ToolTipRole)
            if key == config_manager.get_setting('chat_gpt_api_predefined_context'):
                initial_index = index
            index += 1
        self.predefinedContextBox.currentIndexChanged.connect(self.predefined_context_box_changed)
        self.predefinedContextBox.setCurrentIndex(initial_index)
        # set availability of self.contextEdit in case there is no index changed
        self.contextEdit.setDisabled(True) if initial_index != 1 else self.contextEdit.setEnabled(True)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QFormLayout()
        # https://platform.openai.com/account/api-keys
        chat_after_function_called = config_manager.get_translation('chatAfterFunctionCalled')
        run_python_script_globally = config_manager.get_translation('runPythonScriptGlobally')
        auto_scroll = config_manager.get_translation('autoScroll')
        predefined_context = config_manager.get_translation('predefinedContext')
        context = config_manager.get_translation('chatContext')
        apply_context = config_manager.get_translation('applyContext')
        latest_online_search_result = config_manager.get_translation('latestOnlineSearchResults')
        maximum_online_search_results = config_manager.get_translation('maximumOnlineSearchResults')
        required = config_manager.get_translation('required')
        optional = config_manager.get_translation('optional')
        layout.addRow(f'OpenAI API Key [{required}]:', self.apiKeyEdit)
        layout.addRow(f'Organization ID [{optional}]:', self.orgEdit)
        layout.addRow(f'API Model [{required}]:', self.apiModelBox)
        layout.addRow(f'Max Token [{required}]:', self.maxTokenEdit)
        layout.addRow(f'Function Calling [{optional}]:', self.functionCallingBox)
        layout.addRow(f'{chat_after_function_called} [{optional}]:', self.chatAfterFunctionCalledCheckBox)
        layout.addRow(f'{predefined_context} [{optional}]:', self.predefinedContextBox)
        layout.addRow(f'{context} [{optional}]:', self.contextEdit)
        layout.addRow(f'{apply_context} [{optional}]:', self.applyContextIn)
        layout.addRow(f'{latest_online_search_result} [{optional}]:', self.loadingInternetSearchesBox)
        layout.addRow(f'{maximum_online_search_results} [{optional}]:', self.maxInternetSearchResults)
        layout.addRow(f'{auto_scroll} [{optional}]:', self.autoScrollingCheckBox)
        layout.addRow(f'{run_python_script_globally} [{optional}]:', self.runPythonScriptGloballyCheckBox)
        layout.addWidget(button_box)
        self.autoScrollingCheckBox.stateChanged.connect(self.toggle_auto_scrolling_checkbox)
        self.chatAfterFunctionCalledCheckBox.stateChanged.connect(self.toggle_chat_after_function_called)
        self.runPythonScriptGloballyCheckBox.stateChanged.connect(self.toggle_run_python_script_globally)
        self.functionCallingBox.currentIndexChanged.connect(self.dunction_calling_box_changes)
        self.loadingInternetSearchesBox.currentIndexChanged.connect(self.loading_internet_searches_box_changes)

        # Gemini settings
        self.geminiApiKeyEdit = QLineEdit(config_manager.get_setting('gemini_api_key'))
        self.geminiApiKeyEdit.setEchoMode(QLineEdit.Password)
        self.geminiModelNameEdit = QLineEdit(config_manager.get_setting('gemini_model_name'))

        self.llmProviderBox = QComboBox()
        providers = ["openai", "gemini"]
        current_provider = config_manager.get_setting('selected_llm_provider')
        self.llmProviderBox.addItems(providers)
        if current_provider in providers:
            self.llmProviderBox.setCurrentIndex(providers.index(current_provider))
        else:
            self.llmProviderBox.setCurrentIndex(0) # Default to openai

        # Store references to rows for visibility toggling
        self.openai_model_row_widgets = [self.apiModelBox] # Assuming apiModelBox is for OpenAI
        self.gemini_model_row_widgets = [self.geminiModelNameEdit, self.geminiApiKeyEdit] # Gemini API key is also Gemini-specific here

        # Add LLM Provider and common OpenAI settings
        layout.addRow(f'LLM Provider:', self.llmProviderBox)
        self.openai_api_key_label = QLabel(f'OpenAI API Key [{optional}]:') # Need to access label too
        layout.addRow(self.openai_api_key_label, self.apiKeyEdit)
        self.org_label = QLabel(f'Organization ID [{optional}]:')
        layout.addRow(self.org_label, self.orgEdit)
        self.openai_model_label = QLabel(f'OpenAI API Model [{required}]:')
        layout.addRow(self.openai_model_label, self.apiModelBox)

        # Add Gemini Rows (will be shown/hidden)
        self.gemini_api_key_label = QLabel(f'Gemini API Key [{optional}]:')
        layout.addRow(self.gemini_api_key_label, self.geminiApiKeyEdit)
        self.gemini_model_label = QLabel(f'Gemini Model Name [{optional}]:')
        layout.addRow(self.gemini_model_label, self.geminiModelNameEdit)

        # Connect provider change signal
        self.llmProviderBox.currentIndexChanged.connect(self.update_model_fields_visibility)
        self.update_model_fields_visibility() # Initial call to set correct visibility

        self.setLayout(layout)

    def update_model_fields_visibility(self):
        provider = self.llmProviderBox.currentText()
        is_openai = provider == 'openai'
        is_gemini = provider == 'gemini'

        # Toggle OpenAI specific fields
        self.apiKeyEdit.setVisible(is_openai)
        self.openai_api_key_label.setVisible(is_openai)
        self.orgEdit.setVisible(is_openai)
        self.org_label.setVisible(is_openai)
        self.apiModelBox.setVisible(is_openai)
        self.openai_model_label.setVisible(is_openai)
        # Potentially populate self.apiModelBox here if it's empty and OpenAI is selected
        if is_openai and self.apiModelBox.count() == 0:
             # This is a temporary fix; ideally, the dialog gets the provider instance or model list
            try:
                temp_openai_provider = OpenAIProvider(config_manager) # config_manager is global here
                models = temp_openai_provider.get_available_models()
                self.apiModelBox.addItems(models)
                current_openai_model = config_manager.get_setting('chat_gpt_api_model')
                if current_openai_model in models:
                    self.apiModelBox.setCurrentText(current_openai_model)
            except Exception as e:
                print(f"Error populating OpenAI models: {e}")


        # Toggle Gemini specific fields
        self.geminiApiKeyEdit.setVisible(is_gemini)
        self.gemini_api_key_label.setVisible(is_gemini)
        self.geminiModelNameEdit.setVisible(is_gemini)
        self.gemini_model_label.setVisible(is_gemini)


    def api_key(self) -> str: # This now specifically refers to OpenAI API Key
        return self.apiKeyEdit.text().strip()

    def org(self) -> str:
        return self.orgEdit.text().strip()

    def context(self) -> str:
        return self.contextEdit.text().strip()

    def context_in_all_inputs(self) -> bool:
        return bool(self.applyContextIn.currentIndex() == 1)

    def predefined_context_box_changed(self, index: str) -> bool | None:
        self.contextEdit.setDisabled(True) if index != 1 else self.contextEdit.setEnabled(True)

    def predefined_context(self) -> str:
        return self.predefinedContextBox.currentText()
        # return self.predefinedContextBox.currentData(Qt.ToolTipRole)

    def api_model(self) -> str:
        # return "gpt-3.5-turbo"
        return self.apiModelBox.currentText()

    def function_calling(self) -> str:
        return self.functionCallingBox.currentText()

    def enable_auto_scrolling(self) -> str | bool:
        return self.chatGPTApiAutoScrolling

    def toggle_auto_scrolling_checkbox(self, *, state: bool) -> bool | None:
        self.chatGPTApiAutoScrolling = bool(state)

    def enable_chat_after_function_called(self) -> str | bool:
        return self.chatAfterFunctionCalled

    def toggle_chat_after_function_called(self, *, state: bool) -> bool | None:
        self.chatAfterFunctionCalled = bool(state)

    def enable_run_python_script_globally(self) -> str | bool:
        return self.runPythonScriptGlobally

    def toggle_run_python_script_globally(self, *, state: bool) -> bool | None:
        self.runPythonScriptGlobally = bool(state)

    def dunction_calling_box_changes(self) -> None:
        if self.functionCallingBox.currentText() == 'none' and self.loadingInternetSearchesBox.currentText() == 'auto':
            self.loadingInternetSearchesBox.setCurrentText('none')

    def loading_internet_searches(self) -> str:
        return self.loadingInternetSearchesBox.currentText()

    def loading_internet_searches_box_changes(self) -> None:
        if self.loadingInternetSearchesBox.currentText() == 'auto':
            self.functionCallingBox.setCurrentText('auto')

    def max_token(self) -> str:
        return self.maxTokenEdit.text().strip()

    def max_internet_search_results(self) -> str:
        return self.maxInternetSearchResults.text().strip()

    def gemini_api_key(self) -> str:
        return self.geminiApiKeyEdit.text().strip()

    def gemini_model_name(self) -> str:
        return self.geminiModelNameEdit.text().strip()

    def selected_llm_provider(self) -> str:
        return self.llmProviderBox.currentText()


class Database:
    def __init__(self, file_path: str = '') -> None:
        def regexp(expr: str, item: str) -> bool:
            reg = re.compile(expr, flags=re.IGNORECASE)
            return reg.search(item) is not None

        default_file_path = (
            config_manager.get_setting('chat_gpt_api_last_chat_database')
            if config_manager.get_setting('chat_gpt_api_last_chat_database') and Path(config_manager.get_setting('chat_gpt_api_last_chat_database')).is_file()
            else Path(wd) / 'chats' / 'default.chat'
        )
        self.file_path = file_path or default_file_path
        self.connection = sqlite3.connect(self.file_path)
        self.connection.create_function('REGEXP', 2, regexp)
        self.cursor = self.connection.cursor()
        self.cursor.execute('CREATE TABLE IF NOT EXISTS data (id TEXT PRIMARY KEY, title TEXT, content TEXT)')
        self.connection.commit()

    def insert(self, i_d: str, title: str, content: str) -> None:
        self.cursor.execute('SELECT * FROM data WHERE id = ?', (i_d,))
        existing_data = self.cursor.fetchone()
        if existing_data:
            if existing_data[1] == title and existing_data[2] == content:
                return
            self.cursor.execute('UPDATE data SET title = ?, content = ? WHERE id = ?', (title, content, i_d))
            self.connection.commit()
        else:
            self.cursor.execute('INSERT INTO data (id, title, content) VALUES (?, ?, ?)', (i_d, title, content))
            self.connection.commit()

    def search(self, title: str, content: str) -> list[Any]:
        if config_manager.get_setting('regexp_search_enabled'):
            # with regex
            self.cursor.execute('SELECT * FROM data WHERE title REGEXP ? AND content REGEXP ?', (title, content))
        else:
            # without regex
            self.cursor.execute('SELECT * FROM data WHERE title LIKE ? AND content LIKE ?', (f'%{title}%', f'%{content}%'))
        return self.cursor.fetchall()

    def delete(self, i_d: str) -> None:
        self.cursor.execute('DELETE FROM data WHERE id = ?', (i_d,))
        self.connection.commit()

    def clear(self) -> None:
        self.cursor.execute('DELETE FROM data')
        self.connection.commit()


class ChatGPTAPI(QWidget):
    def __init__(self, parent: Any) -> None:
        super().__init__()
        # config.chatGPTApi = self
        self.parent = parent

        # LLM Provider will be initialized here
        self.llm_provider: LLMProvider = None
        self.initialize_llm_provider()

        # set variables
        self.setup_variables()
        # run plugins
        # self.runPlugins()
        # setup interface
        self.setup_ui()
        # load database
        self.load_data()
        # new entry at launch
        self.new_data()

    def open_database(self) -> None:
        # Show a file dialog to get the file path to open
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Database', os.path.join(wd, 'chats', 'default.chat'), 'ChatGPTQT Database (*.chat)', options=options)

        # If the user selects a file path, open the file
        self.database = Database(file_path)
        self.load_data()
        self.update_title(file_path)
        self.new_data()

    def new_database(self, copy_existing: bool = False) -> None:
        # Show a file dialog to get the file path to save
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'New Database',
            os.path.join(wd, 'chats', self.database.file_path if copy_existing else 'new.chat'),
            'ChatGPTQT Database (*.chat)',
            options=options,
        )

        # If the user selects a file path, save the file
        if file_path:
            # make sure the file ends with ".chat"
            if not file_path.endswith('.chat'):
                file_path += '.chat'
            # ignore if copy currently opened database
            if copy_existing and Path(file_path).resolve() == Path(self.database.file_path).resolve():
                return
            # Check if the file already exists
            if Path(file_path).is_file():
                # Ask the user if they want to replace the existing file
                msg_box = QMessageBox()
                msg_box.setWindowTitle('Confirm overwrite')
                msg_box.setText(f'The file {file_path} already exists. Do you want to replace it?')
                msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg_box.setDefaultButton(QMessageBox.No)
                if msg_box.exec() == QMessageBox.No:
                    return
                Path(file_path).unlink()

            # create a new database
            if copy_existing:
                shutil.copy(self.database.file_path, file_path)
            self.database = Database(file_path)
            self.load_data()
            self.update_title(file_path)
            self.new_data()

    def update_title(self, file_path: str | Path = '') -> None:
        if not file_path:
            file_path = self.database.file_path
        config_manager.update_setting('chat_gpt_api_last_chat_database', str(file_path))
        basename = Path(file_path).name
        self.parent.setWindowTitle(f'ChatGPTQT - {basename}')

    def setup_variables(self) -> None:
        self.busyLoading = False
        self.contentID = ''
        self.database = Database()
        self.update_title()
        self.data_list = []
        # self.recognitionThread = SpeechRecognitionThread(self)
        # self.recognitionThread.phrase_recognized.connect(self.onPhraseRecognized)

    def setup_ui(self) -> None:  # noqa: PLR0914, PLR0915
        lt0 = QHBoxLayout()
        self.setLayout(lt0)
        widget_lt = QWidget()
        lt0l = QVBoxLayout()
        widget_lt.setLayout(lt0l)
        wdr = QWidget()
        lt0r = QVBoxLayout()
        wdr.setLayout(lt0r)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(widget_lt)
        splitter.addWidget(wdr)
        lt0.addWidget(splitter)

        # widgets on the right
        self.searchInput = QLineEdit()
        self.searchInput.setClearButtonEnabled(True)
        self.replaceInput = QLineEdit()
        self.replaceInput.setClearButtonEnabled(True)
        self.userInput = QLineEdit()
        completer = QCompleter(config_manager.get_setting('input_suggestions'))
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.userInput.setCompleter(completer)
        self.userInput.setPlaceholderText(config_manager.get_translation('messageHere'))
        self.userInput.mousePressEvent = lambda _: self.userInput.selectAll()
        self.userInput.setClearButtonEnabled(True)
        self.userInputMultiline = QPlainTextEdit()
        self.userInputMultiline.setPlaceholderText(config_manager.get_translation('messageHere'))
        self.voiceCheckbox = QCheckBox(config_manager.get_translation('voice'))
        self.voiceCheckbox.setToolTip(config_manager.get_translation('voiceTyping'))
        self.voiceCheckbox.setCheckState(Qt.Unchecked)
        self.contentView = QPlainTextEdit()
        self.contentView.setReadOnly(True)
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 0)  # Set the progress bar to use an indeterminate progress indicator
        api_key_button = QPushButton(config_manager.get_translation('settings'))
        self.multilineButton = QPushButton('+')
        font_metrics = QFontMetrics(self.multilineButton.font())
        text_rect = font_metrics.boundingRect(self.multilineButton.text())
        button_width = text_rect.width() + 20
        button_height = text_rect.height() + 10
        self.multilineButton.setFixedSize(button_width, button_height)
        self.sendButton = QPushButton(config_manager.get_translation('send'))
        search_label = QLabel(config_manager.get_translation('searchFor'))
        replace_label = QLabel(config_manager.get_translation('replaceWith'))
        search_replace_button = QPushButton(config_manager.get_translation('replace'))
        search_replace_button.setToolTip(config_manager.get_translation('replaceSelectedText'))
        search_replace_button_all = QPushButton(config_manager.get_translation('all'))
        search_replace_button_all.setToolTip(config_manager.get_translation('replaceAll'))
        self.apiModels = QComboBox()
        self.apiModels.addItems([config_manager.get_translation('chat'), config_manager.get_translation('image'), 'browser', 'python', 'system'])
        self.apiModels.setCurrentIndex(0)
        self.apiModel = 0
        self.newButton = QPushButton(config_manager.get_translation('new'))
        save_button = QPushButton(config_manager.get_translation('save'))
        self.editableCheckbox = QCheckBox(config_manager.get_translation('editable'))
        self.editableCheckbox.setCheckState(Qt.Unchecked)
        # self.audioCheckbox = QCheckBox(config_manager.get_setting("this_translation").audio)
        # self.audioCheckbox.setCheckState(Qt.Checked if config_manager.get_setting("chat_gpt_api_audio") else Qt.Unchecked)
        self.choiceNumber = QComboBox()
        self.choiceNumber.addItems([str(i) for i in range(1, 11)])
        self.choiceNumber.setCurrentIndex(config_manager.get_setting('chat_gpt_api_no_of_choices') - 1)
        self.fontSize = QComboBox()
        self.fontSize.addItems([str(i) for i in range(1, 51)])
        self.fontSize.setCurrentIndex(config_manager.get_setting('font_size') - 1)
        self.temperature = QComboBox()
        self.temperature.addItems([str(i / 10) for i in range(21)])
        self.temperature.setCurrentIndex(int(config_manager.get_setting('chat_gpt_api_temperature') * 10))
        temperature_label = QLabel(config_manager.get_translation('temperature'))
        temperature_label.setAlignment(Qt.AlignRight)
        temperature_label.setToolTip('''What sampling temperature to use, between 0 and 2.
Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.''')
        choices_label = QLabel(config_manager.get_translation('choices'))
        choices_label.setAlignment(Qt.AlignRight)
        choices_label.setToolTip('How many chat completion choices to generate for each input message.')
        font_label = QLabel(config_manager.get_translation('font'))
        font_label.setAlignment(Qt.AlignRight)
        font_label.setToolTip(config_manager.get_translation('fontSize'))
        prompt_layout = QHBoxLayout()
        user_input_layout = QVBoxLayout()
        user_input_layout.addWidget(self.userInput)
        user_input_layout.addWidget(self.userInputMultiline)
        self.userInputMultiline.hide()
        prompt_layout.addLayout(user_input_layout)
        # if sphinx_installed:
        #     prompt_layout.addWidget(self.voiceCheckbox)
        prompt_layout.addWidget(self.multilineButton)
        prompt_layout.addWidget(self.sendButton)
        prompt_layout.addWidget(self.apiModels)
        lt0r.addLayout(prompt_layout)
        lt0r.addWidget(self.contentView)
        lt0r.addWidget(self.progressBar)
        self.progressBar.hide()
        search_replace_layout = QHBoxLayout()
        search_replace_layout.addWidget(search_label)
        search_replace_layout.addWidget(self.searchInput)
        search_replace_layout.addWidget(replace_label)
        search_replace_layout.addWidget(self.replaceInput)
        search_replace_layout.addWidget(search_replace_button)
        search_replace_layout.addWidget(search_replace_button_all)
        lt0r.addLayout(search_replace_layout)
        rt_control_layout = QHBoxLayout()
        rt_control_layout.addWidget(api_key_button)
        rt_control_layout.addWidget(temperature_label)
        rt_control_layout.addWidget(self.temperature)
        rt_control_layout.addWidget(choices_label)
        rt_control_layout.addWidget(self.choiceNumber)
        rt_control_layout.addWidget(font_label)
        rt_control_layout.addWidget(self.fontSize)
        rt_control_layout.addWidget(self.editableCheckbox)
        # rtControlLayout.addWidget(self.audioCheckbox)
        rt_button_layout = QHBoxLayout()
        rt_button_layout.addWidget(self.newButton)
        rt_button_layout.addWidget(save_button)
        lt0r.addLayout(rt_control_layout)
        lt0r.addLayout(rt_button_layout)

        # widgets on the left
        help_button = QPushButton(config_manager.get_translation('help'))
        search_title_button = QPushButton(config_manager.get_translation('searchTitle'))
        search_content_button = QPushButton(config_manager.get_translation('searchContent'))
        self.searchTitle = QLineEdit()
        self.searchTitle.setClearButtonEnabled(True)
        self.searchTitle.setPlaceholderText(config_manager.get_translation('searchTitleHere'))
        self.searchContent = QLineEdit()
        self.searchContent.setClearButtonEnabled(True)
        self.searchContent.setPlaceholderText(config_manager.get_translation('searchContentHere'))
        self.listView = QListView()
        self.listModel = QStandardItemModel()
        self.listView.setModel(self.listModel)
        remove_button = QPushButton(config_manager.get_translation('remove'))
        clear_all_button = QPushButton(config_manager.get_translation('clearAll'))
        search_title_layout = QHBoxLayout()
        search_title_layout.addWidget(self.searchTitle)
        search_title_layout.addWidget(search_title_button)
        lt0l.addLayout(search_title_layout)
        search_content_layout = QHBoxLayout()
        search_content_layout.addWidget(self.searchContent)
        search_content_layout.addWidget(search_content_button)
        lt0l.addLayout(search_content_layout)
        lt0l.addWidget(self.listView)
        lt_button_layout = QHBoxLayout()
        lt_button_layout.addWidget(remove_button)
        lt_button_layout.addWidget(clear_all_button)
        lt0l.addLayout(lt_button_layout)
        lt0l.addWidget(help_button)

        # Connections
        self.userInput.returnPressed.connect(self.send_message)
        help_button.clicked.connect(lambda: webbrowser.open('https://github.com/iroedius/ChatGPTQT/wiki'))
        api_key_button.clicked.connect(self.show_api_dialog)
        self.multilineButton.clicked.connect(self.multiline_button_clicked)
        self.sendButton.clicked.connect(self.send_message)
        save_button.clicked.connect(self.save_data)
        self.newButton.clicked.connect(self.new_data)
        search_title_button.clicked.connect(self.search_data)
        search_content_button.clicked.connect(self.search_data)
        self.searchTitle.textChanged.connect(self.search_data)
        self.searchContent.textChanged.connect(self.search_data)
        self.listView.clicked.connect(self.select_data)
        clear_all_button.clicked.connect(self.clear_data)
        remove_button.clicked.connect(self.remove_data)
        self.editableCheckbox.stateChanged.connect(self.toggle_editable)
        # self.audioCheckbox.stateChanged.connect(self.toggleChatGPTApiAudio)
        # self.voiceCheckbox.stateChanged.connect(self.toggleVoiceTyping)
        self.choiceNumber.currentIndexChanged.connect(self.update_choise_number)
        self.apiModels.currentIndexChanged.connect(self.update_api_model)
        self.fontSize.currentIndexChanged.connect(self.set_font_size)
        self.temperature.currentIndexChanged.connect(self.update_temperature)
        search_replace_button.clicked.connect(self.replace_selected_text)
        search_replace_button_all.clicked.connect(self.search_replace_all)
        self.searchInput.returnPressed.connect(self.search_chat_content)
        self.replaceInput.returnPressed.connect(self.replace_selected_text)

        self.set_font_size()
        self.update_search_tool_tips()

    def set_font_size(self, index: int | None = None) -> None:
        if index is not None:
            config_manager.update_setting('font_size', index + 1)
        # content view
        font = self.contentView.font()
        font.setPointSize(config_manager.get_setting('font_size'))
        self.contentView.setFont(font)
        # list view
        font = self.listView.font()
        font.setPointSize(config_manager.get_setting('font_size'))
        self.listView.setFont(font)

    def update_search_tool_tips(self) -> None:
        if config_manager.get_setting('regexp_search_enabled'):
            self.searchTitle.setToolTip(config_manager.get_translation('matchingRegularExpression'))
            self.searchContent.setToolTip(config_manager.get_translation('matchingRegularExpression'))
            self.searchInput.setToolTip(config_manager.get_translation('matchingRegularExpression'))
        else:
            self.searchTitle.setToolTip('')
            self.searchContent.setToolTip('')
            self.searchInput.setToolTip('')

    def search_chat_content(self) -> None:
        search = QRegularExpression(self.searchInput.text()) if config_manager.get_setting('regexp_search_enabled') else self.searchInput.text()
        self.contentView.find(search)

    def replace_selected_text(self) -> None:
        current_selected_text = self.contentView.textCursor().selectedText()
        if current_selected_text:
            search_input = self.searchInput.text()
            replace_input = self.replaceInput.text()
            if search_input:
                replace = (
                    re.sub(search_input, replace_input, current_selected_text)
                    if config_manager.get_setting('regexp_search_enabled')
                    else current_selected_text.replace(search_input, replace_input)
                )
            else:
                replace = self.replaceInput.text()
            self.contentView.insertPlainText(replace)

    def search_replace_all(self) -> None:
        search = self.searchInput.text()
        if search:
            replace = self.replaceInput.text()
            content = self.contentView.toPlainText()
            new_content = re.sub(search, replace, content, flags=re.MULTILINE) if config_manager.get_setting('regexp_search_enabled') else content.replace(search, replace)
            self.contentView.setPlainText(new_content)

    def multiline_button_clicked(self) -> None:
        if self.userInput.isVisible():
            self.userInput.hide()
            self.userInputMultiline.setPlainText(self.userInput.text())
            self.userInputMultiline.show()
            self.multilineButton.setText('-')
        else:
            self.userInputMultiline.hide()
            self.userInput.setText(self.userInputMultiline.toPlainText())
            self.userInput.show()
            self.multilineButton.setText('+')
        self.set_uset_input_focus()

    def set_uset_input_focus(self) -> None:
        self.userInput.setFocus() if self.userInput.isVisible() else self.userInputMultiline.setFocus()

    def show_api_dialog(self) -> None:
        dialog = ApiDialog(self)
        result = dialog.exec()
        if result == QDialog.Accepted:
            config_manager.update_setting('openai_api_key', dialog.api_key())
            if not openai.api_key: # This should also use the snake_case key
                openai.api_key = os.environ['OPENAI_API_KEY'] = config_manager.get_setting('openai_api_key')
            config_manager.update_setting('openai_api_organization', dialog.org())
            try:
                max_tokens = int(dialog.max_token())
                min_value = 20
                if max_tokens < min_value:
                    max_tokens = min_value
                config_manager.update_setting('chat_gpt_api_max_tokens', max_tokens)
            except ValueError:
                logging.exception('Invalid value for chatGPTApiMaxTokens')
            try:
                max_results = int(dialog.max_internet_search_results())
                max_value = 100
                if max_results <= 0:
                    max_results = 1
                elif max_results > max_value:
                    max_results = max_value
                config_manager.update_setting('maximum_internet_search_results', max_results)
            except ValueError:
                logging.exception('Invalid value for maximumInternetSearchResults')

            config_manager.update_setting('chat_gpt_api_auto_scrolling', dialog.enable_auto_scrolling())
            config_manager.update_setting('run_python_script_globally', dialog.enable_run_python_script_globally())
            config_manager.update_setting('chat_after_function_called', dialog.enable_chat_after_function_called())

            # Save model based on selected provider
            selected_provider = dialog.selected_llm_provider()
            if selected_provider == 'openai':
                config_manager.update_setting('chat_gpt_api_model', dialog.api_model())
            elif selected_provider == 'gemini':
                config_manager.update_setting('gemini_model_name', dialog.gemini_model_name())

            config_manager.update_setting('chat_gpt_api_function_call', dialog.function_calling()) # This is OpenAI specific
            config_manager.update_setting('loading_internet_searches', dialog.loading_internet_searches())

            internet_searches = 'integrate google searches'
            exclude_list = config_manager.get_setting('chat_gpt_plugin_exclude_list')
            current_loading_behavior = config_manager.get_setting('loading_internet_searches')

            if current_loading_behavior == 'auto':
                if internet_searches in exclude_list:
                    exclude_list.remove(internet_searches)
                    config_manager.update_setting('chat_gpt_plugin_exclude_list', exclude_list)
                    self.parent.reloadMenubar()
            elif current_loading_behavior == 'none':
                if internet_searches not in exclude_list:
                    exclude_list.append(internet_searches)
                    config_manager.update_setting('chat_gpt_plugin_exclude_list', exclude_list)
                    self.parent.reloadMenubar()

            self.run_plugins()
            config_manager.update_setting('chat_gpt_api_predefined_context', dialog.predefined_context())
            config_manager.update_setting('chat_gpt_api_context_in_all_inputs', dialog.context_in_all_inputs())
            config_manager.update_setting('chat_gpt_api_context', dialog.context())

            # Save Gemini settings
            config_manager.update_setting('gemini_api_key', dialog.gemini_api_key())
            config_manager.update_setting('gemini_model_name', dialog.gemini_model_name())
            config_manager.update_setting('selected_llm_provider', dialog.selected_llm_provider())

            # config_manager.get_setting("chat_gpt_api_audio_language") = dialog.language()
            self.new_data()
            # Potentially reload or re-initialize parts of the UI if LLM provider changes
            self.initialize_llm_provider() # Re-initialize provider on settings change
            # self.parent.setWindowTitle(f"ChatGPTQT - LLM: {config_manager.get_setting('selected_llm_provider').upper()}") # initialize_llm_provider will do this


    def initialize_llm_provider(self) -> None:
        provider_name = self.config_manager.get_setting('selected_llm_provider')
        if provider_name == 'gemini':
            self.llm_provider = GeminiProvider(self.config_manager)
        elif provider_name == 'openai':
            self.llm_provider = OpenAIProvider(self.config_manager)
        else:
            # Default to OpenAI if setting is invalid or not set
            self.config_manager.update_setting('selected_llm_provider', 'openai')
            self.llm_provider = OpenAIProvider(self.config_manager)

        # self.llm_provider.load_config() # Already called in provider's __init__

        # Update window title
        if self.parent: # Ensure parent (MainWindow) is set
             self.parent.setWindowTitle(f"ChatGPTQT - LLM: {self.config_manager.get_setting('selected_llm_provider').upper()}")


    def update_api_model(self, index: int) -> None:
        self.apiModel = index

    def update_temperature(self, index: int) -> None:
        config_manager.update_setting('chat_gpt_api_temperature', float(index / 10))

    def update_choise_number(self, index: int) -> None:
        config_manager.update_setting('chat_gpt_api_no_of_choices', index + 1)

    def on_phrase_recognized(self, phrase: str) -> None:
        self.userInput.setText(f'{self.userInput.text()} {phrase}')

    # def toggleVoiceTyping(self, state) -> None:
    #     self.recognitionThread.start() if state else self.recognitionThread.stop()

    def toggle_editable(self, state: bool) -> None:
        self.contentView.setReadOnly(not state)

    def toggle_chatgpt_api_audio(self, state: bool) -> None:
        config_manager.update_setting('chat_gpt_api_audio', state)
        if not config_manager.get_setting('chat_gpt_api_audio'):
            self.close_media_player()

    def no_text_selection(self) -> None:
        self.display_message('This feature works on text selection. Select text first!')

    def validate_url(self, url: str) -> bool:
        try:
            result = urllib.parse.urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def web_browse(self, user_input: str = '') -> None:
        if not user_input:
            user_input = self.contentView.textCursor().selectedText().strip()
        if not user_input:
            self.no_text_selection()
            return
        if self.validate_url(user_input):
            url = user_input
        else:
            user_input = urllib.parse.quote(user_input)
            url = f'https://www.google.com/search?q={user_input}'
        webbrowser.open(url)

    def display_text(self, text: str) -> None:
        self.save_data()
        self.new_data()
        self.contentView.setPlainText(text)

    def run_system_command(self, command: str = '') -> None:
        if not command:
            command = self.contentView.textCursor().selectedText().strip()
        if command:
            command = repr(command)
            command = literal_eval(command).replace('\u2029', '\n')
        else:
            self.no_text_selection()
            return

        # display output only, without error
        # output = subprocess.check_output(command, shell=True, text=True)
        # self.displayText(output)

        # display both output and error
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)  # noqa: S602
        output = result.stdout  # Captured standard output
        error = result.stderr  # Captured standard error
        self.display_text(f'> {command}')
        self.contentView.appendPlainText(f'\n{output}')
        if error.strip():
            self.contentView.appendPlainText('\n# Error\n')
            self.contentView.appendPlainText(error)

    def run_python_command(self, command: str = '') -> None:
        if not command:
            command = self.contentView.textCursor().selectedText().strip()
        if command:
            command = repr(command)
            command = literal_eval(command).replace('\u2029', '\n')
        else:
            self.no_text_selection()
            return

        # Store the original standard output
        original_stdout = sys.stdout
        # Create a StringIO object to capture the output
        output = StringIO()
        try:
            # Redirect the standard output to the StringIO object
            sys.stdout = output
            # Execute the Python string in global namespace
            try:
                exec(command, globals()) if config_manager.get_setting('run_python_script_globally') else exec(command)
                captured_output = output.getvalue()
            except Exception:
                captured_output = traceback.format_exc()
            # Get the captured output
        finally:
            # Restore the original standard output
            sys.stdout = original_stdout

        # Display the captured output
        if captured_output.strip():
            self.display_text(captured_output)
        else:
            self.display_message('Done!')

    def remove_data(self) -> None:
        index = self.listView.selectedIndexes()
        if not index:
            return
        confirm = QMessageBox.question(
            self,
            config_manager.get_translation('remove'),
            config_manager.get_translation('areyousure'),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm == QMessageBox.Yes:
            item = index[0]
            data = item.data(Qt.UserRole)
            self.database.delete(data[0])
            self.load_data()
            self.new_data()

    def clear_data(self) -> None:
        confirm = QMessageBox.question(
            self,
            config_manager.get_translation('clearAll'),
            config_manager.get_translation('areyousure'),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm == QMessageBox.Yes:
            self.database.clear()
            self.load_data()

    def save_data(self) -> None:
        text = self.contentView.toPlainText().strip()
        if text:
            lines = text.split('\n')
            if not self.contentID:
                self.contentID = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            title = re.sub('^>>> ', '', lines[0][:50])
            content = text
            self.database.insert(self.contentID, title, content)
            self.load_data()

    def load_data(self) -> None:
        # reverse the list, so that the latest is on the top
        self.data_list = self.database.search('', '')
        if self.data_list:
            self.data_list.reverse()
        self.listModel.clear()
        for data in self.data_list:
            item = QStandardItem(data[1])
            item.setToolTip(data[0])
            item.setData(data, Qt.UserRole)
            self.listModel.appendRow(item)

    def search_data(self) -> None:
        keyword1 = self.searchTitle.text().strip()
        keyword2 = self.searchContent.text().strip()
        self.data_list = self.database.search(keyword1, keyword2)
        self.listModel.clear()
        for data in self.data_list:
            item = QStandardItem(data[1])
            item.setData(data, Qt.UserRole)
            self.listModel.appendRow(item)

    def chat_action(self, context: str = '') -> None:
        if context:
            config_manager.update_setting('chat_gpt_api_predefined_context', context)
        current_selected_text = self.contentView.textCursor().selectedText().strip()
        if current_selected_text:
            self.new_data()
            self.userInput.setText(current_selected_text)
            self.send_message()

    def new_data(self) -> None:
        if not self.busyLoading:
            self.contentID = ''
            self.contentView.setPlainText(self.get_initial_message())
            self.set_uset_input_focus()

    def get_initial_message(self) -> str:
        provider = config_manager.get_setting('selected_llm_provider')
        if provider == 'openai':
            if config_manager.get_setting('openai_api_key'):
                return '' # API key present, no special message needed
            return '''OpenAI API Key is NOT Found!

Follow the following steps:
1) Register and get your OpenAI Key at https://platform.openai.com/account/api-keys
2) Click the "Settings" button below and enter your own OpenAI API key'''
        elif provider == 'gemini':
            if config_manager.get_setting('gemini_api_key'):
                return '' # API key present, no special message needed
            return '''Gemini API Key is NOT Found!

Follow the following steps:
1) Enable the Gemini API in your Google Cloud project & create an API key.
   (Search for "Vertex AI Studio" -> "APIs & Services" -> "Enable APIs and Services" -> search for "Generative Language API" and enable it. Then create credentials.)
2) Click the "Settings" button below and enter your Gemini API key.'''
        return "No LLM provider selected or provider unknown. Please check settings."

    def select_data(self, index: QModelIndex) -> None:
        if not self.busyLoading:
            data = index.data(Qt.UserRole)
            self.contentID = data[0]
            content = data[2]
            self.contentView.setPlainText(content)
            self.set_uset_input_focus()

    def print_data(self) -> None:
        # Get the printer and print dialog
        printer = QPrinter()
        dialog = QPrintDialog(printer, self)

        # If the user clicked "OK" in the print dialog, print the text
        if dialog.exec() == QPrintDialog.Accepted:
            document = QTextDocument()
            document.setPlainText(self.contentView.toPlainText())
            document.print_(printer)

    def export_data(self) -> None:
        # Show a file dialog to get the file path to save
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Export Chat Content', os.path.join(wd, 'chats', 'chat.txt'), 'Text Files (*.txt);;Python Files (*.py);;All Files (*)', options=options,
        )

        # If the user selects a file path, save the file
        if file_path:
            with Path(file_path).open('w', encoding='utf-8') as file_obj:
                file_obj.write(self.contentView.toPlainText().strip())

    def open_text_file_dialog(self) -> None:
        options = QFileDialog.Options()
        file_name, filtr = QFileDialog.getOpenFileName(self, 'Open Text File', 'Text File', 'Plain Text Files (*.txt);;Python Scripts (*.py);;All Files (*)', '', options)
        if file_name:
            with Path(file_name).open('r', encoding='utf-8') as file_obj:
                self.display_text(file_obj.read())

    def display_message(self, message: str = '', title: str = 'ChatGPTQT') -> None:
        QMessageBox.information(self, title, message)

    # The following method was modified from source:
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    # This method is OpenAI specific and uses tiktoken.
    # It would need to be moved to OpenAIProvider or a new method added to LLMProvider abstraction
    # if token counting is desired for all providers.
    # For now, commenting it out from ChatGPTAPI.
    # def num_tokens_from_messages(self, model: str = '') -> int | None:
    #     if not model:
    #         # This should get the model for the *current* LLM provider
    #         if self.llm_provider and hasattr(self.llm_provider, 'model_name'):
    #             model = self.llm_provider.model_name
    #         else: # Fallback or error if provider/model not set
    #             print("Error: LLM Provider or model not set for token counting.")
    #             return None

    #     # Ensure this is only called if it's an OpenAI model, or adapt for Gemini if it has similar token counting
    #     if not (model.startswith("gpt-") or model.startswith("text-davinci-")): # Basic check
    #         print(f"Token counting not implemented for model: {model}")
    #         return None

    #     user_input = self.userInput.text().strip()
    #     # get_messages was removed, need chat_history and current prompt
    #     # This part needs significant rework to fit the new structure if re-enabled.
    #     # messages = self.get_messages(user_input) # Old call
    #     chat_history = self.get_chat_history_from_view()
    #     messages_for_token_count = chat_history + [{'role': 'user', 'content': user_input}]


    #     '''Return the number of tokens used by a list of messages.'''
    #     try:
    #         encoding = tiktoken.encoding_for_model(model)
    #     except KeyError:
    #         print('Warning: model not found. Using cl100k_base encoding.')
    #         encoding = tiktoken.get_encoding('cl100k_base')
    #     # encoding = tiktoken.get_encoding("cl100k_base")
    #     if model in {'gpt-4-turbo-preview', 'gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106'}:
    #         tokens_per_message = 3
    #         tokens_per_name = 1
    #     elif model in {'gpt-4', 'gpt-4-0613'}: # Add other models as needed
    #         tokens_per_message = 3
    #         tokens_per_name = 1
    #     # elif model.startswith("text-davinci-"):
    #     #     tokens_per_message = 0 # Different for older completion models
    #     #     tokens_per_name = 0
    #     else:
    #         msg = (
    #             f'''num_tokens_from_messages() is not implemented for model {model}.
    #             See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.'''
    #         )
    #         # raise NotImplementedError(msg) # Avoid raising error for now
    #         print(msg)
    #         return None

    #     num_tokens = 0
    #     for message in messages_for_token_count:
    #         num_tokens += tokens_per_message
    #         for key, value in message.items():
    #             if value: # Ensure value is not None
    #                 num_tokens += len(encoding.encode(str(value)))
    #             if key == 'name': # Function calling specific
    #                 num_tokens += tokens_per_name
    #     num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    #     self.display_message(message=f'{num_tokens} prompt tokens counted for model {model} (OpenAI).')
    #     return None

    # get_context is largely provider-specific or handled by the provider's _prepare_messages or equivalent.
    # For now, we simplify it or assume the provider fetches its own context if necessary.
    # The main `get_chat_history_from_view` will just extract the raw conversation.
    # def get_context(self) -> str: ... (Removed for now, provider specific)

    def get_chat_history_from_view(self) -> list[dict]:
        """
        Extracts chat history from the contentView in the common format:
        [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        """
        history_text = self.contentView.toPlainText().strip()
        history = []
        if history_text:
            # Remove the initial prompt if it's there (e.g. from OpenAI key message)
            # This logic might need refinement based on how initial messages are structured.
            # For now, assuming actual chat starts after first ">>>" if any.
            if not history_text.startswith(">>> "):
                 # If no ">>>", it could be a single assistant message (e.g. error) or initial state.
                 # For history purposes, we can't parse this into user/assistant turns easily without ">>>".
                 # So, if no ">>>", assume no parseable history for this simplified version.
                 pass # Or, treat the whole thing as an initial assistant message if that makes sense for some flows.

            exchanges = [exchange for exchange in history_text.split('\n>>> ') if exchange.strip()]
            first_exchange = True
            for exchange in exchanges:
                if first_exchange and history_text.startswith(">>> "): # remove the first ">>> "
                    exchange = exchange[4:]
                    first_exchange = False

                parts = exchange.split('\n~~~ ', 1) # Split only on the first occurrence of ~~~
                user_content = parts[0].strip()
                history.append({'role': 'user', 'content': user_content})
                if len(parts) > 1:
                    assistant_content = parts[1].strip()
                    # Further split assistant content if multiple responses were recorded (e.g. "### Response 2:")
                    # For now, taking the whole block as one assistant turn for simplicity in history.
                    # Actual display of multiple responses is handled by print/process_response.
                    history.append({'role': 'assistant', 'content': assistant_content})
        return history

    def print(self, text: str) -> None:
        self.contentView.appendPlainText(f'\n{text}' if self.contentView.toPlainText() else text)
        self.contentView.setPlainText(re.sub('\n\n[\n]+?([^\n])', r'\n\n\1', self.contentView.toPlainText()))

    # FIXME: transformers
    def print_stream(self, text: str) -> None:
        # transform responses
        # for t in config_manager.get_setting('chat_gpt_transformers'):
        #     text = t(text)
        self.contentView.setPlainText(self.contentView.toPlainText() + text)
        # no audio for streaming tokens
        # if config_manager.get_setting("chat_gpt_api_audio"):
        #    self.play_audio(text)
        # scroll to the bottom
        if config_manager.get_setting('chat_gpt_api_auto_scrolling'):
            content_scroll_bar = self.contentView.verticalScrollBar()
            content_scroll_bar.setValue(content_scroll_bar.maximum())

    def send_message(self) -> None:
        if self.userInputMultiline.isVisible():
            self.multiline_button_clicked()
        if self.apiModel == 0:
            self.get_response()
        elif self.apiModel == 1:
            self.get_image()
        elif self.apiModel == 2:
            user_input = self.userInput.text().strip()
            if user_input:
                self.web_browse(user_input)
        elif self.apiModel == 3:
            user_input = self.userInput.text().strip()
            if user_input:
                self.run_python_command(user_input)
        elif self.apiModel == 4:
            user_input = self.userInput.text().strip()
            if user_input:
                self.run_system_command(user_input)

    def get_image(self) -> None:
        if not self.progressBar.isVisible():
            user_input = self.userInput.text().strip()
            if user_input:
                self.userInput.setDisabled(True)
                self.progressBar.show()  # show progress bar
                OpenAIImage(self).work_on_get_response(user_input)

    def display_image(self, image_url: str) -> None:
        if image_url:
            webbrowser.open(image_url)
            self.userInput.setEnabled(True)
            self.progressBar.hide()

    def get_response(self) -> None:
        if self.progressBar.isVisible() and config_manager.get_setting('chat_gpt_api_no_of_choices') == 1:
            stop_file = '.stop_chatgpt'
            if not Path(stop_file).is_file():
                Path(stop_file).open('a').close()
        elif not self.progressBar.isVisible():
            user_input = self.userInput.text().strip()
            if user_input:
                self.userInput.setDisabled(True)
                if config_manager.get_setting('chat_gpt_api_no_of_choices') == 1:
                    self.sendButton.setText(config_manager.get_translation('stop'))
                    self.busyLoading = True
                    self.listView.setDisabled(True)
                    self.newButton.setDisabled(True)

                # Prepare history and prompt for the LLM provider
                chat_history = self.get_chat_history_from_view()
                # The user_input is the current new prompt

                self.print(f'>>> {user_input}') # Display the new prompt
                self.save_data() # Save current state including the new prompt
                self.currentLoadingID = self.contentID
                self.currentLoadingContent = self.contentView.toPlainText().strip()
                self.progressBar.show()  # show progress bar
                ChatGPTResponse(self).work_on_get_response(messages)  # get chatGPT response in a separate thread

    def file_names_without_extension(self, _dir: str, ext: str) -> None:
        files = glob.glob(os.path.join(_dir, f'*.{ext}'))
        return sorted([file[len(_dir) + 1 : -(len(ext) + 1)] for file in files if os.path.isfile(file)])

    def exec_python_file(self, script) -> None:
        if config_manager.get_setting('developer'): # developer is already snake_case
            with open(script, encoding='utf8') as f:
                code = compile(f.read(), script, 'exec')
                exec(code, globals())
        else:
            try:
                with open(script, encoding='utf8') as f:
                    code = compile(f.read(), script, 'exec')
                    exec(code, globals())
            except Exception:
                msg = f'Failed to run "{Path(script).name}"!'
                logging.exception(msg)
                print(msg)

    def run_plugins(self) -> None:
        # The settings 'predefined_contexts', 'input_suggestions',
        # 'chat_gpt_transformers', 'chat_gpt_api_function_signatures',
        # and 'chat_gpt_api_available_functions' are now initialized by ConfigManager
        # with their defaults if not present in settings.json.
        # Plugins can still modify them via config_manager.update_setting if needed.

        plugin_folder = Path.cwd() / 'plugins'
        # always run 'integrate google searches'
        internet_searches = 'integrate google searches'
        script = plugin_folder / f'{internet_searches}.py'
        self.exec_python_file(script, self) # Pass self (ChatGPTAPI instance)
        for plugin in self.file_names_without_extension(plugin_folder, 'py'):
            if plugin != internet_searches and plugin not in config_manager.get_setting('chat_gpt_plugin_exclude_list'):
                script = plugin_folder / f'{plugin}.py'
                self.exec_python_file(script, self) # Pass self
        # if internetSeraches in config_manager.get_setting("chat_gpt_plugin_exclude_list"):
        #     config_manager.update_setting("chat_gpt_api_function_signatures[0], ''") # Example, if needed

    # FIXME: transformers
    def process_response(self, responses: str) -> None:
        if responses:
            # reload the working content in case users change it during waiting for response
            self.contentID = self.currentLoadingID
            self.contentView.setPlainText(self.currentLoadingContent)
            self.currentLoadingID = self.currentLoadingContent = ''
            # transform responses
            # for t in config_manager.get_setting('chat_gpt_transformers'):
            #     responses = t(responses)
            # update new reponses
            self.print(responses)
            # scroll to the bottom
            if config_manager.get_setting('chat_gpt_api_auto_scrolling'):
                content_scroll_bar = self.contentView.verticalScrollBar()
                content_scroll_bar.setValue(content_scroll_bar.maximum())
            # if not (responses.startswith("OpenAI API re") or responses.startswith("Failed to connect to OpenAI API:")) and config_manager.get_setting("chat_gpt_api_audio"):
            #        self.play_audio(responses)
        # empty user input
        self.userInput.setText('')
        # auto-save
        self.save_data()
        # hide progress bar
        self.userInput.setEnabled(True)
        if config_manager.get_setting('chat_gpt_api_no_of_choices') == 1:
            self.listView.setEnabled(True)
            self.newButton.setEnabled(True)
            self.busyLoading = False
        self.sendButton.setText(config_manager.get_translation('send'))
        self.progressBar.hide()
        self.set_uset_input_focus()

    def play_audio(self, responses: str) -> None:
        text_list = [i.replace('>>>', '').strip() for i in responses.split('\n') if i.strip()]
        audio_files = []
        for index, text in enumerate(text_list):
            with contextlib.suppress(Exception):
                audio_file = (Path('temp') / f'gtts_{index}.mp3').resolve()
                if audio_file.is_file():
                    audio_file.unlink()
                gTTS(text=text, lang=config_manager.get_setting('chat_gpt_api_audio_language') or 'en').save(
                    audio_file,
                )
                audio_files.append(audio_file)
        if audio_files:
            self.play_audio_files(audio_files)

    def play_audio_files(self, files: list[str]) -> None:
        pass

    def close_media_player(self) -> None:
        pass


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.init_ui()

    def reload_menubar(self) -> None:
        self.menuBar().clear()
        self.create_menubar()

    def create_menubar(self) -> None:
        # Create a menu bar
        menubar = self.menuBar()

        # Create a File menu and add it to the menu bar
        file_menu = menubar.addMenu(config_manager.get_translation('chat'))

        new_action = QAction(config_manager.get_translation('openDatabase'), self)
        new_action.setShortcut('Ctrl+Shift+O')
        new_action.triggered.connect(self.chatGPT.open_database)
        file_menu.addAction(new_action)

        new_action = QAction(config_manager.get_translation('newDatabase'), self)
        new_action.setShortcut('Ctrl+Shift+N')
        new_action.triggered.connect(self.chatGPT.new_database)
        file_menu.addAction(new_action)

        new_action = QAction(config_manager.get_translation('saveDatabaseAs'), self)
        new_action.setShortcut('Ctrl+Shift+S')
        new_action.triggered.connect(lambda: self.chatGPT.new_database(copy_existing=True))
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        new_action = QAction(config_manager.get_translation('fileManager'), self)
        new_action.triggered.connect(self.open_database_directory)
        file_menu.addAction(new_action)

        new_action = QAction(config_manager.get_translation('pluginDirectory'), self)
        new_action.triggered.connect(self.open_plugins_directory)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        new_action = QAction(config_manager.get_translation('newChat'), self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.chatGPT.new_data)
        file_menu.addAction(new_action)

        new_action = QAction(config_manager.get_translation('saveChat'), self)
        new_action.setShortcut('Ctrl+S')
        new_action.triggered.connect(self.chatGPT.save_data)
        file_menu.addAction(new_action)

        new_action = QAction(config_manager.get_translation('exportChat'), self)
        new_action.triggered.connect(self.chatGPT.export_data)
        file_menu.addAction(new_action)

        new_action = QAction(config_manager.get_translation('printChat'), self)
        new_action.setShortcut('Ctrl+P')
        new_action.triggered.connect(self.chatGPT.print_data)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        new_action = QAction(config_manager.get_translation('readTextFile'), self)
        new_action.triggered.connect(self.chatGPT.open_text_file_dialog)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        new_action = QAction(config_manager.get_translation('countPromptTokens'), self)
        new_action.triggered.connect(self.chatGPT.num_tokens_from_messages)
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        # Create a Exit action and add it to the File menu
        exit_action = QAction(config_manager.get_translation('exit'), self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip(config_manager.get_translation('exitTheApplication'))
        exit_action.triggered.connect(QGuiApplication.instance().quit)
        file_menu.addAction(exit_action)

        # Create customise menu
        customise_menu = menubar.addMenu(config_manager.get_translation('customise'))

        open_settings = QAction(config_manager.get_translation('configure'), self)
        open_settings.triggered.connect(self.chatGPT.show_api_dialog)
        customise_menu.addAction(open_settings)

        customise_menu.addSeparator()

        new_action = QAction(config_manager.get_translation('toggleDarkTheme'), self)
        new_action.triggered.connect(self.toggle_theme)
        customise_menu.addAction(new_action)

        new_action = QAction(config_manager.get_translation('toggleSystemTray'), self)
        new_action.triggered.connect(self.toggle_system_tray)
        customise_menu.addAction(new_action)

        new_action = QAction(config_manager.get_translation('toggleMultilineInput'), self)
        new_action.setShortcut('Ctrl+L')
        new_action.triggered.connect(self.chatGPT.multiline_button_clicked)
        customise_menu.addAction(new_action)

        new_action = QAction(config_manager.get_translation('toggleRegexp'), self)
        new_action.setShortcut('Ctrl+E')
        new_action.triggered.connect(self.toggle_regexp)
        customise_menu.addAction(new_action)

        # Create predefined context menu
        context_menu = menubar.addMenu(config_manager.get_translation('predefinedContext'))
        # Assuming get_setting('predefined_contexts') returns a dictionary or a string to be evaluated
        predefined_contexts_menu = config_manager.get_setting('predefined_contexts')
        if isinstance(predefined_contexts_menu, str):
            try:
                predefined_contexts_menu = literal_eval(predefined_contexts_menu)
            except (ValueError, SyntaxError):
                predefined_contexts_menu = {} # Default to empty if parsing fails

        for index, context in enumerate(predefined_contexts_menu.keys()): # Iterate over keys
            context_action = QAction(context, self)
            if index < 10:
                context_action.setShortcut(f'Ctrl+{index}')
            context_action.triggered.connect(partial(self.chatGPT.chat_action, context))
            context_menu.addAction(context_action)

        # TODO:
        # Create a plugin menu
        # plugin_menu = menubar.addMenu(config_manager.get_translation('plugins'))

        # plugin_folder = Path.cwd() / 'plugins'
        # for _, plugin in enumerate(self.file_names_without_extension(plugin_folder, 'py')):
        #     new_action = QAction(plugin, self)
        #     new_action.setCheckable(True)
        #     new_action.setChecked(bool(plugin in config_manager.get_setting('chatGPTPluginExcludeList')))
        #     new_action.triggered.connect(partial(self.update_exclude_plugin_list, plugin))
        #     plugin_menu.addAction(new_action)

        # Create a text selection menu
        text_selection_menu = menubar.addMenu(config_manager.get_translation('textSelection'))

        new_action = QAction(config_manager.get_translation('webBrowser'), self)
        new_action.triggered.connect(self.chatGPT.web_browse)
        text_selection_menu.addAction(new_action)

        new_action = QAction(config_manager.get_translation('runAsPythonCommand'), self)
        new_action.triggered.connect(self.chatGPT.run_python_command)
        text_selection_menu.addAction(new_action)

        new_action = QAction(config_manager.get_translation('runAsSystemCommand'), self)
        new_action.triggered.connect(self.chatGPT.run_system_command)
        text_selection_menu.addAction(new_action)

        # Create About menu
        about_menu = menubar.addMenu(config_manager.get_translation('about'))

        open_settings = QAction(config_manager.get_translation('repository'), self)
        open_settings.triggered.connect(lambda: webbrowser.open('https://github.com/iroedius/ChatGPTQT'))
        about_menu.addAction(open_settings)

        about_menu.addSeparator()

        new_action = QAction(config_manager.get_translation('help'), self)
        new_action.triggered.connect(lambda: webbrowser.open('https://github.com/iroedius/ChatGPTQT/wiki'))
        about_menu.addAction(new_action)

    def init_ui(self) -> None:
        # Set a central widget
        self.chatGPT = ChatGPTAPI(self)
        self.setCentralWidget(self.chatGPT)

        # create menu bar
        self.create_menubar()

        # set initial window size
        # self.setWindowTitle("ChatGPTQT")
        self.resize(QGuiApplication.primaryScreen().availableSize() * (3 / 4))
        self.show()

    def update_exclude_plugin_list(self, plugin: str) -> None:
        exclude_list = config_manager.get_setting('chat_gpt_plugin_exclude_list')
        if plugin in exclude_list:
            exclude_list.remove(plugin)
        else:
            exclude_list.append(plugin)
        config_manager.update_setting('chat_gpt_plugin_exclude_list', exclude_list)

        internet_searches = 'integrate google searches'
        if internet_searches in exclude_list and config_manager.get_setting('loading_internet_searches') == 'auto':
            config_manager.update_setting('loading_internet_searches', 'none')
        elif internet_searches not in exclude_list and config_manager.get_setting('loading_internet_searches') == 'none':
            config_manager.update_setting('loading_internet_searches', 'auto')
            config_manager.update_setting('chat_gpt_api_function_call', 'auto')
        # reload plugins
        self.chatGPT.run_plugins()

    def file_names_without_extension(self, directory: Path, ext: str) -> list:
        return [item for item in directory.glob(f'*.{ext}') if item.is_file()]

    def get_open_command(self) -> str:
        this_os = platform.system()
        open_command = ''
        if this_os == 'Windows':
            open_command = 'start'
        elif this_os == 'Darwin':
            open_command = 'open'
        elif this_os == 'Linux':
            open_command = 'xdg-open'
        return open_command

    def open_database_directory(self) -> None:
        database_directory = Path(config_manager.get_setting('chat_gpt_api_last_chat_database')).parent
        open_command = self.get_open_command()
        os.system(f'{open_command} {database_directory}')

    def open_plugins_directory(self) -> None:
        open_command = self.get_open_command()
        os.system(f'{open_command} plugins')

    def toggle_regexp(self) -> None:
        current_value = config_manager.get_setting('regexp_search_enabled')
        new_value = not current_value
        config_manager.update_setting('regexp_search_enabled', new_value)
        self.chatGPT.update_search_tool_tips()
        QMessageBox.information(self, 'ChatGPTQT', f"Regex for search and replace is {'enabled' if new_value else 'disabled'}!")

    def toggle_system_tray(self) -> None:
        current_value = config_manager.get_setting('enable_system_tray')
        new_value = not current_value
        config_manager.update_setting('enable_system_tray', new_value)
        QMessageBox.information(self, 'ChatGPTQT', 'You need to restart this application to make the changes effective.')

    def toggle_theme(self) -> None:
        is_dark_theme = config_manager.get_setting('dark_theme')
        config_manager.update_setting('dark_theme', not is_dark_theme) # This was correct
        qdarktheme.setup_theme() if not is_dark_theme else qdarktheme.setup_theme('light') # This logic was correct

    # Work with system tray
    def is_wayland(self) -> bool:
        return bool(platform.system() == 'Linux' and os.getenv('QT_QPA_PLATFORM') is not None and os.getenv('QT_QPA_PLATFORM') == 'wayland')

    def bring_to_foreground(self, window: QMainWindow) -> None:
        if window and not (window.isVisible() and window.isActiveWindow()):
            window.raise_()
            # Method activateWindow() does not work with qt.qpa.wayland
            # platform.system() == "Linux" and not os.getenv('QT_QPA_PLATFORM') is None and os.getenv('QT_QPA_PLATFORM') == "wayland"
            # The error message is received when QT_QPA_PLATFORM=wayland:
            # qt.qpa.wayland: Wayland does not support QWindow::requestActivate()
            # Therefore, we use hide and show methods instead with wayland.
            if window.isVisible() and not window.isActiveWindow():
                window.hide()
            window.show()
            if not self.is_wayland():
                window.activateWindow()


if __name__ == '__main__':
    main_window_instance = None

    def show_main_window() -> None:
        global main_window_instance
        if not main_window_instance:
            main_window_instance = MainWindow()
            qdarktheme.setup_theme() if config_manager.get_setting('dark_theme') else qdarktheme.setup_theme('light')
        else:
            main_window_instance.bring_to_foreground(main_window_instance)

    # TODO: probably unneeded
    def about_to_quit() -> None:
        pass
        # with Path('config.py').open('w', encoding='utf-8') as file_obj:
        #     for name in dir(config):
        #         exclude_from_saving_list = (
        #             'mainWindow',  # main window object
        #             'chatGPTApi',  # GUI object
        #             'chatGPTTransformers',  # used with plugins; transform ChatGPT response message
        #             'predefinedContexts',  # used with plugins; pre-defined contexts
        #             'inputSuggestions',  # used with plugins; user input suggestions
        #             'integrate_google_searches_signature',
        #             'chatGPTApiFunctionSignatures',  # used with plugins; function calling
        #             'chatGPTApiAvailableFunctions',  # used with plugins; function calling
        #             'pythonFunctionResponse',  # used with plugins; function calling when function name is 'python'
        #         )
        #         if not name.startswith('__') and name not in exclude_from_saving_list:
        #             try:
        #                 value = config_manager.get_setting(name)
        #                 file_obj.write(f'{name} = {pprint.pformat(value)}\n')
        #                 config_manager.update_setting(name, value)
        #             except Exception:
        #                 logging.exception('Exception while saving settings to config file')

    platform = platform.system()
    app_name = 'ChatGPTQT'
    icon_path = str(Path().cwd().resolve() / 'icons' / f'{app_name}.ico')
    # Windows icon
    if platform == 'Windows':
        myappid = 'chatgptqt'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(icon_path)
    # app
    app = QApplication(sys.argv)
    app_icon = QIcon(icon_path)
    app.setWindowIcon(app_icon)
    show_main_window()
    # connection
    app.aboutToQuit.connect(about_to_quit)

    # Desktop shortcut
    # on Windows
    if platform == 'Windows':
        desktop_path = Path.home() / 'Desktop'
        shortcut_dir = desktop_path if desktop_path.is_dir() else wd
        shortcut_bat = shortcut_dir / f'{app_name}.bat'
        shortcut_command = f'''powershell.exe -NoExit -Command "python '{this_file}'"'''
        # Create .bat for application shortcuts
        if not shortcut_bat.exists():
            with contextlib.suppress(Exception):
                shortcut_bat.write_text(shortcut_command)
    # on macOS
    elif platform == 'Darwin':
        shortcut_file = Path(f'~/Desktop/{app_name}.command').expanduser()
        if not shortcut_file.is_file():
            with shortcut_file.open('w') as f:
                f.write('#!/bin/bash\n')
                f.write(f'cd {wd}\n')
                f.write(f'{sys.executable} {this_file} gui\n')
            Path(shortcut_file).chmod(0o755)
    # additional shortcuts on Linux
    elif platform == 'Linux':
        def desktop_file_content() -> str:
            icon_path = wd / 'icons' / 'ChatGPTQT.png'
            return f'''#!/usr/bin/env xdg-open

[Desktop Entry]
Version=1.0
Type=Application
Terminal=false
Path={wd}
Exec={sys.executable} {this_file}
Icon={icon_path}
Name=ChatGPTQT
'''

        linux_desktop_file = Path(wd) / f'{app_name}.desktop'
        if not linux_desktop_file.exists():
            # Create .desktop shortcut
            with contextlib.suppress(Exception):
                linux_desktop_file.write_text(desktop_file_content(), encoding=locale.getpreferredencoding(do_setlocale=False))
                # Try to copy the newly created .desktop file to:
                # ~/.local/share/applications
                user_app_dir = Path.home() / '.local' / 'share' / 'applications'
                user_app_dir_shortcut = user_app_dir / f'{app_name}.desktop'
                user_app_dir.mkdir(parents=True, exist_ok=True)
                if not user_app_dir_shortcut.exists():
                    copyfile(linux_desktop_file, user_app_dir_shortcut)
                # ~/Desktop
                desktop_path = Path(os.environ['HOME']) / 'Desktop'
                desktop_path_shortcut = desktop_path / f'{app_name}.desktop'
                if not desktop_path_shortcut.is_file():
                    copyfile(linux_desktop_file, desktop_path_shortcut)

    # system tray
    if config_manager.get_setting('enable_system_tray'):
        app.setQuitOnLastWindowClosed(False)
        # Set up tray icon
        tray = QSystemTrayIcon()
        tray.setIcon(app_icon)
        tray.setToolTip('ChatGPTQT')
        tray.setVisible(True)
        # Import system tray menu
        tray_menu = QMenu()
        show_main_window_action = QAction(config_manager.get_translation('show'))
        show_main_window_action.triggered.connect(show_main_window)
        tray_menu.addAction(show_main_window_action)
        # Add a separator
        tray_menu.addSeparator()
        # Quit
        quit_app_action = QAction(config_manager.get_translation('exit'))
        quit_app_action.triggered.connect(app.quit)
        tray_menu.addAction(quit_app_action)
        tray.setContextMenu(tray_menu)

    # run the app
    sys.exit(app.exec())
