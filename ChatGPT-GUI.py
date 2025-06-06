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
# from qtmodern import styles, windows # Removed qtmodern
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

config_manager = ConfigManager()

this_file = Path(__file__).resolve()
wd = this_file.parent
if Path.cwd() != wd:
    os.chdir(wd)
if not Path('config.py').is_file():
    Path('config.py').open('a').close()

DARK_THEME_QSS = """
QWidget {
    background-color: #2e2e2e;
    color: #e0e0e0;
    font-family: "Segoe UI", Arial, sans-serif; /* Example font */
}

QMainWindow {
    background-color: #2e2e2e;
}

QLabel {
    color: #e0e0e0;
    background-color: transparent; /* Ensure labels don't have their own background unless specified */
}

QPushButton {
    background-color: #4a4a4a;
    color: #e0e0e0;
    border: 1px solid #555555;
    padding: 6px 12px;
    border-radius: 4px;
    min-width: 80px; /* Ensure buttons have a decent minimum width */
}
QPushButton:hover {
    background-color: #5a5a5a;
    border: 1px solid #666666;
}
QPushButton:pressed {
    background-color: #3a3a3a;
}
QPushButton:disabled {
    background-color: #404040;
    color: #888888;
    border-color: #454545;
}

QLineEdit, QPlainTextEdit {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 5px;
    selection-background-color: #5a6f8c; /* A bluish selection color */
    selection-color: #e0e0e0;
}
QLineEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #77aaff; /* Highlight focus */
}
QLineEdit:read-only, QPlainTextEdit:read-only {
    background-color: #333333;
    color: #a0a0a0;
}


QListView {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555555;
    border-radius: 4px;
    alternate-background-color: #424242; /* For alternating row colors */
    outline: 0; /* Remove focus outline if not desired */
}
QListView::item {
    padding: 5px;
}
QListView::item:selected {
    background-color: #5a6f8c; /* Selection color */
    color: #ffffff;
}
QListView::item:hover {
    background-color: #4f4f4f; /* Hover color */
}


QComboBox {
    background-color: #4a4a4a;
    color: #e0e0e0;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 5px;
    min-width: 6em;
}
QComboBox:editable {
    background-color: #3c3c3c; /* Background of the line edit part */
}
QComboBox:hover {
    background-color: #5a5a5a;
    border-color: #666666;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 1px;
    border-left-color: #555555;
    border-left-style: solid;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
    background-color: #4a4a4a;
}
QComboBox::down-arrow {
    image: url(icons/down_arrow_light.png); /* Placeholder: Needs an actual icon */
}
QComboBox::down-arrow:on { /* Arrow when combo box is open */
    /* image: url(icons/up_arrow_light.png); */ /* Placeholder */
}
QComboBox QAbstractItemView { /* Style for the dropdown list */
    background-color: #3c3c3c;
    border: 1px solid #555555;
    selection-background-color: #5a6f8c;
    color: #e0e0e0;
}

QCheckBox {
    spacing: 5px;
    color: #e0e0e0;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #555555;
    border-radius: 3px;
    background-color: #3c3c3c;
}
QCheckBox::indicator:unchecked:hover {
    border: 1px solid #666666;
}
QCheckBox::indicator:checked {
    background-color: #5a6f8c; /* Check mark color */
    border: 1px solid #5a6f8c;
    image: url(icons/checkmark_light.png); /* Placeholder: Needs an actual checkmark icon */
}
QCheckBox::indicator:checked:hover {
    background-color: #6a7f9c;
    border: 1px solid #6a7f9c;
}
QCheckBox::indicator:disabled {
    border: 1px solid #454545;
    background-color: #404040;
}


QMenuBar {
    background-color: #383838;
    color: #e0e0e0;
    border-bottom: 1px solid #2a2a2a; /* Separator line */
}
QMenuBar::item {
    background-color: transparent;
    padding: 5px 10px;
}
QMenuBar::item:selected { /* When hovered */
    background-color: #4f4f4f;
}
QMenuBar::item:pressed { /* When menu is open */
    background-color: #5a5a5a;
}

QMenu {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #505050;
    padding: 5px;
}
QMenu::item {
    padding: 5px 20px 5px 20px; /* Top, Right, Bottom, Left */
    border-radius: 3px;
}
QMenu::item:selected {
    background-color: #5a6f8c;
    color: #ffffff;
}
QMenu::separator {
    height: 1px;
    background-color: #505050;
    margin-left: 10px;
    margin-right: 5px;
}
QMenu::indicator { /* For checkable QAction */
    width: 13px;
    height: 13px;
    /* padding-left: 5px; */ /* Align with text */
}


QProgressBar {
    border: 1px solid #555555;
    border-radius: 4px;
    text-align: center;
    color: #e0e0e0;
    background-color: #3c3c3c;
}
QProgressBar::chunk {
    background-color: #5a6f8c;
    width: 10px; /* Width of the progress segments */
    margin: 0.5px;
}

QScrollBar:vertical {
    border: 1px solid #3a3a3a;
    background: #2e2e2e;
    width: 15px;
    margin: 15px 0 15px 0; /* Top, Right, Bottom, Left - leave space for arrows */
    border-radius: 0px;
}
QScrollBar::handle:vertical {
    background-color: #4f4f4f;
    min-height: 20px;
    border-radius: 3px;
    border: 1px solid #555555;
}
QScrollBar::handle:vertical:hover {
    background-color: #5a5a5a;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: 1px solid #3a3a3a;
    background: #4a4a4a;
    height: 14px;
    subcontrol-origin: margin;
}
QScrollBar::add-line:vertical:hover, QScrollBar::sub-line:vertical:hover {
    background: #5a5a5a;
}
QScrollBar::add-line:vertical {
    subcontrol-position: bottom;
}
QScrollBar::sub-line:vertical {
    subcontrol-position: top;
}
/* TODO: Add up/down arrow images for scrollbar */

QScrollBar:horizontal {
    border: 1px solid #3a3a3a;
    background: #2e2e2e;
    height: 15px;
    margin: 0 15px 0 15px; /* Top, Right, Bottom, Left - leave space for arrows */
    border-radius: 0px;
}
QScrollBar::handle:horizontal {
    background-color: #4f4f4f;
    min-width: 20px;
    border-radius: 3px;
    border: 1px solid #555555;
}
QScrollBar::handle:horizontal:hover {
    background-color: #5a5a5a;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    border: 1px solid #3a3a3a;
    background: #4a4a4a;
    width: 14px;
    subcontrol-origin: margin;
}
QScrollBar::add-line:horizontal:hover, QScrollBar::sub-line:horizontal:hover {
    background: #5a5a5a;
}
QScrollBar::add-line:horizontal {
    subcontrol-position: right;
}
QScrollBar::sub-line:horizontal {
    subcontrol-position: left;
}
/* TODO: Add left/right arrow images for scrollbar */


QSplitter::handle {
    background-color: #3a3a3a;
    border: 1px solid #2a2a2a;
    width: 5px; /* Vertical splitter */
    height: 5px; /* Horizontal splitter */
}
QSplitter::handle:hover {
    background-color: #4f4f4f;
}
QSplitter::handle:pressed {
    background-color: #5a5a5a;
}

QToolTip {
    background-color: #2e2e2e;
    color: #e0e0e0;
    border: 1px solid #555555;
    padding: 4px;
    opacity: 230; /* Slightly transparent */
}

QDialog {
    background-color: #2e2e2e;
}

QDialogButtonBox QPushButton { /* Ensure buttons in dialogs also get styled */
    min-width: 80px;
}

QTabBar::tab {
    background: #3c3c3c;
    border: 1px solid #555555;
    border-bottom-color: #3c3c3c; /* Same as background color for selected tab */
    padding: 8px 15px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    color: #a0a0a0;
}

QTabBar::tab:selected, QTabBar::tab:hover {
    background: #4a4a4a;
    color: #e0e0e0;
}

QTabBar::tab:selected {
    border-color: #555555;
    border-bottom-color: #4a4a4a; /* Make selected tab blend with content area */
}

QTabWidget::pane { /* The area where tab pages are shown */
    border: 1px solid #555555;
    background: #4a4a4a; /* Should match selected tab background */
}

QStatusBar {
    background-color: #383838;
    color: #e0e0e0;
    border-top: 1px solid #2a2a2a;
}

/* Placeholder for icons - actual icons would need to be provided */
/* For example, QComboBox::down-arrow, QCheckBox::indicator:checked, QScrollBar arrows */
/* You might need to create/find these icons and place them in an 'icons' directory */
/* QComboBox::down-arrow { image: url(./icons/arrow_down_dark.svg); } */
/* QCheckBox::indicator:checked { image: url(./icons/checkbox_checked_dark.svg); } */

"""


class SpeechRecognitionThread(QThread):
    phrase_recognized = Signal(str)

    def __init__(self, parent: QThread) -> None:
        super().__init__(parent)
        self.is_running = False

    def run(self) -> None:
        self.is_running = True
        # if config_manager.get_setting('pocketsphinxModelPath'):
        #     # download English dictionary at: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
        #     # download voice models at https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/
        #     speech = LiveSpeech(
        #         # sampling_rate=16000,  # optional
        #         hmm=get_model_path(config_manager.get_setting('pocketsphinxModelPath')),
        #         lm=get_model_path(config_manager.get_setting('pocketsphinxModelPathBin')),
        #         dic=get_model_path(config_manager.get_setting('pocketsphinxModelPathDict')),
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
        for key, value in config_manager.get_internal('predefinedContexts').items():
            self.predefinedContextBox.addItem(key)
            self.predefinedContextBox.setItemData(self.predefinedContextBox.count() - 1, value, role=Qt.ToolTipRole)
            if key == config_manager.get_setting('chatGPTApiPredefinedContext'):
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

        self.setLayout(layout)

    def api_key(self) -> str:
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


class Database:
    def __init__(self, file_path: str = '') -> None:
        def regexp(expr: str, item: str) -> bool:
            reg = re.compile(expr, flags=re.IGNORECASE)
            return reg.search(item) is not None

        default_file_path = (
            config_manager.get_setting('chatGPTApiLastChatDatabase')
            if config_manager.get_setting('chatGPTApiLastChatDatabase') and Path(config_manager.get_setting('chatGPTApiLastChatDatabase')).is_file()
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
        if config_manager.get_setting('regexpSearchEnabled'):
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
        config_manager.update_internal('chatGPTApi', self)
        self.parent = parent
        # required
        openai.api_key = os.environ['OPENAI_API_KEY'] = config_manager.get_setting('openaiApiKey')
        # optional
        if config_manager.get_setting('openaiApiOrganization'):
            openai.organization = config_manager.get_setting('openaiApiOrganization')
        # set title
        self.setWindowTitle('ChatGPTQT')
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
        config_manager.update_setting('chatGPTApiLastChatDatabase', str(file_path))
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
        completer = QCompleter(config_manager.get_setting('inputSuggestions'))
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
        # self.audioCheckbox.setCheckState(Qt.Checked if config_manager.get_setting("chatGPTApiAudio") else Qt.Unchecked)
        self.choiceNumber = QComboBox()
        self.choiceNumber.addItems([str(i) for i in range(1, 11)])
        self.choiceNumber.setCurrentIndex(int(config_manager.get_setting('chatGPTApiNoOfChoices')) - 1)
        self.fontSize = QComboBox()
        self.fontSize.addItems([str(i) for i in range(1, 51)])
        self.fontSize.setCurrentIndex(int(config_manager.get_setting('fontSize')) - 1)
        self.temperature = QComboBox()
        self.temperature.addItems([str(i / 10) for i in range(21)])
        self.temperature.setCurrentIndex(int(config_manager.get_setting('chatGPTApiTemperature') * 10))
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
            config_manager.update_setting('fontSize', str(index + 1))
        # content view
        font = self.contentView.font()
        font.setPointSize(int(config_manager.get_setting('fontSize')))
        self.contentView.setFont(font)
        # list view
        font = self.listView.font()
        font.setPointSize(int(config_manager.get_setting('fontSize')))
        self.listView.setFont(font)

    def update_search_tool_tips(self) -> None:
        if config_manager.get_setting('regexpSearchEnabled'):
            self.searchTitle.setToolTip(config_manager.get_translation('matchingRegularExpression'))
            self.searchContent.setToolTip(config_manager.get_translation('matchingRegularExpression'))
            self.searchInput.setToolTip(config_manager.get_translation('matchingRegularExpression'))
        else:
            self.searchTitle.setToolTip('')
            self.searchContent.setToolTip('')
            self.searchInput.setToolTip('')

    def search_chat_content(self) -> None:
        search = QRegularExpression(self.searchInput.text()) if config_manager.get_setting('regexpSearchEnabled') else self.searchInput.text()
        self.contentView.find(search)

    def replace_selected_text(self) -> None:
        current_selected_text = self.contentView.textCursor().selectedText()
        if current_selected_text:
            search_input = self.searchInput.text()
            replace_input = self.replaceInput.text()
            if search_input:
                replace = (
                    re.sub(search_input, replace_input, current_selected_text)
                    if config_manager.get_setting('regexpSearchEnabled')
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
            new_content = re.sub(search, replace, content, flags=re.MULTILINE) if config_manager.get_setting('regexpSearchEnabled') else content.replace(search, replace)
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
            config_manager.update_setting('openaiApiKey', dialog.api_key())
            if not openai.api_key:
                openai.api_key = os.environ['OPENAI_API_KEY'] = config_manager.get_setting('openaiApiKey')
            config_manager.update_setting('openaiApiOrganization', dialog.org())
            try:
                config_manager.update_setting('chatGPTApiMaxTokens', dialog.max_token())
                min_value = 20
                if int(config_manager.get_setting('chatGPTApiMaxTokens')) < min_value:
                    config_manager.update_setting('chatGPTApiMaxTokens', str(min_value))
            except Exception:
                logging.exception('Exception while updating chatGPTApiMaxTokens')
            try:
                config_manager.update_setting('maximumInternetSearchResults', dialog.max_internet_search_results())
                max_value = 100
                if int(config_manager.get_setting('maximumInternetSearchResults')) <= 0:
                    config_manager.update_setting('maximumInternetSearchResults', '1')
                elif int(config_manager.get_setting('maximumInternetSearchResults')) > max_value:
                    config_manager.update_setting('maximumInternetSearchResults', str(max_value))
            except Exception:
                logging.exception('Exception while getting maximumInternetSearchResults')
            # config_manager.update_setting("includeDuckDuckGoSearchResults") = dialog.include_internet_searches()
            config_manager.update_setting('chatGPTApiAutoScrolling', dialog.enable_auto_scrolling())
            config_manager.update_setting('runPythonScriptGlobally', dialog.enable_run_python_script_globally())
            config_manager.update_setting('chatAfterFunctionCalled', dialog.enable_chat_after_function_called())
            config_manager.update_setting('chatGPTApiModel', dialog.api_model())
            config_manager.update_setting('chatGPTApiFunctionCall', dialog.function_calling())
            config_manager.update_setting('loadingInternetSearches', dialog.loading_internet_searches())
            internet_searches = 'integrate google searches'
            if config_manager.get_setting('loadingInternetSearches') == 'auto' and internet_searches in config_manager.get_setting('chatGPTPluginExcludeList'):
                config_manager.update_setting('chatGPTPluginExcludeList.remove', internet_searches)
                self.parent.reloadMenubar()
            elif config_manager.get_setting('loadingInternetSearches') == 'none' and internet_searches not in config_manager.get_setting(chatGPTPluginExcludeList):
                config_manager.update_setting('chatGPTPluginExcludeList.append', internet_searches)
                self.parent.reloadMenubar()
            self.run_plugins()
            config_manager.update_setting('chatGPTApiPredefinedContext', dialog.predefined_context())
            config_manager.update_setting('chatGPTApiContextInAllInputs', dialog.context_in_all_inputs())
            config_manager.update_setting('chatGPTApiContext', dialog.context())
            # config_manager.get_setting("chatGPTApiAudioLanguage") = dialog.language()
            self.new_data()

    def update_api_model(self, index: int) -> None:
        self.apiModel = index

    def update_temperature(self, index: int) -> None:
        config_manager.update_setting('chatGPTApiTemperature', float(index / 10))

    def update_choise_number(self, index: int) -> None:
        config_manager.update_setting('chatGPTApiNoOfChoices', index + 1)

    def on_phrase_recognized(self, phrase: str) -> None:
        self.userInput.setText(f'{self.userInput.text()} {phrase}')

    # def toggleVoiceTyping(self, state) -> None:
    #     self.recognitionThread.start() if state else self.recognitionThread.stop()

    def toggle_editable(self, state: bool) -> None:
        self.contentView.setReadOnly(not state)

    def toggle_chatgpt_api_audio(self, state: bool) -> None:
        config_manager.update_setting('chatGPTApiAudio', state)
        if not config_manager.get_setting('chatGPTApiAudio'):
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
                exec(command, globals()) if config_manager.get_setting('runPythonScriptGlobally') else exec(command)
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
            config_manager.update_setting('chatGPTApiPredefinedContext', context)
        current_selected_text = self.contentView.textCursor().selectedText().strip()
        if current_selected_text:
            self.new_data()
            self.userInput.setText(current_selected_text)
            self.send_message()

    def new_data(self) -> None:
        if not self.busyLoading:
            self.contentID = ''
            self.contentView.setPlainText(
                ''
                if openai.api_key
                else '''OpenAI API Key is NOT Found!

Follow the following steps:
1) Register and get your OpenAI Key at https://platform.openai.com/account/api-keys
2) Click the "Settings" button below and enter your own OpenAI API key''',
            )
            self.set_uset_input_focus()

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
    def num_tokens_from_messages(self, model: str = '') -> int | None:
        if not model:
            model = config_manager.get_setting('chatGPTApiModel')
        user_input = self.userInput.text().strip()
        messages = self.get_messages(user_input)

        '''Return the number of tokens used by a list of messages.'''
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print('Warning: model not found. Using cl100k_base encoding.')
            encoding = tiktoken.get_encoding('cl100k_base')
        # encoding = tiktoken.get_encoding("cl100k_base")
        if model in {'gpt-4-turbo-preview', 'gpt-4-0125-preview', 'gpt-4-1106-preview', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-1106'}:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model in {'gpt-4', 'gpt-4-0613'}:
            # print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.num_tokens_from_messages(messages, model='gpt-4-0613')
        else:
            msg = (
                f'''num_tokens_from_messages() is not implemented for model {model}.
                See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.'''
            )
            raise NotImplementedError(
                msg,
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == 'name':
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        # return num_tokens
        self.display_message(message=f'{num_tokens} prompt tokens counted!')
        return None

    def get_context(self) -> str:
        if config_manager.get_setting('chatGPTApiPredefinedContext') not in config_manager.get_setting('predefinedContexts'):
            config_manager.update_setting('chatGPTApiPredefinedContext', '[none]')
        if config_manager.get_setting('chatGPTApiPredefinedContext') == '[none]':
            # no context
            context = ''
        elif config_manager.get_setting('chatGPTApiPredefinedContext') == '[custom]':
            # custom input in the settings dialog
            context = config_manager.get_setting('chatGPTApiContext')
        else:
            # users can modify config_manager.get_setting("predefinedContexts") via plugins
            context = config_manager.get_internal('predefinedContexts')[config_manager.get_setting('chatGPTApiPredefinedContext')]
            # change config for particular contexts
            if config_manager.get_setting('chatGPTApiPredefinedContext') == 'Execute Python Code':
                if config_manager.get_setting('chatGPTApiFunctionCall') == 'none':
                    config_manager.get_setting('chatGPTApiFunctionCall', 'auto')
                if config_manager.get_setting('loadingInternetSearches') == 'always':
                    config_manager.get_setting('loadingInternetSearches', 'auto')
        return context

    def get_messages(self, user_input: str) -> list:
        # system message
        system_message = "You're a kind helpful assistant."
        if config_manager.get_setting('chatGPTApiFunctionCall') == 'auto' and config_manager.get_setting('chatGPTApiFunctionSignatures'):
            system_message += ' Only use the functions you have been provided with.'
        messages = [{'role': 'system', 'content': system_message}]
        # predefine context
        context = self.get_context()
        # chat history
        history = self.contentView.toPlainText().strip()
        if history:
            if (
                context
                and config_manager.get_setting('chatGPTApiPredefinedContext') != 'Execute Python Code'
                and not config_manager.get_setting('chatGPTApiContextInAllInputs')
            ):
                messages.append({'role': 'assistant', 'content': context})
            if history.startswith('>>> '):
                history = history[4:]
            exchanges = [exchange for exchange in history.split('\n>>> ') if exchange.strip()]
            for exchange in exchanges:
                qa = exchange.split('\n~~~ ')
                for i, content in enumerate(qa):
                    if i == 0:
                        messages.append({'role': 'user', 'content': content.strip()})
                    else:
                        messages.append({'role': 'assistant', 'content': content.strip()})
        # customise chat context
        if context and (
            config_manager.get_setting('chatGPTApiPredefinedContext') == 'Execute Python Code'
            or (not history or (history and config_manager.get_setting('chatGPTApiContextInAllInputs')))
        ):
            # messages.append({"role": "assistant", "content": context})
            user_input = f'{context}\n{user_input}'
        # user input
        messages.append({'role': 'user', 'content': user_input})
        return messages

    def print(self, text: str) -> None:
        self.contentView.appendPlainText(f'\n{text}' if self.contentView.toPlainText() else text)
        self.contentView.setPlainText(re.sub('\n\n[\n]+?([^\n])', r'\n\n\1', self.contentView.toPlainText()))

    # FIXME: transformers
    def print_stream(self, text: str) -> None:
        # transform responses
        # for t in config_manager.get_internal('chatGPTTransformers'):
        #     text = t(text)
        self.contentView.setPlainText(self.contentView.toPlainText() + text)
        # no audio for streaming tokens
        # if config_manager.get_setting("chatGPTApiAudio"):
        #    self.play_audio(text)
        # scroll to the bottom
        if config_manager.get_setting('chatGPTApiAutoScrolling'):
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
        if self.progressBar.isVisible() and config_manager.get_setting('chatGPTApiNoOfChoices') == 1:
            stop_file = '.stop_chatgpt'
            if not Path(stop_file).is_file():
                Path(stop_file).open('a').close()
        elif not self.progressBar.isVisible():
            user_input = self.userInput.text().strip()
            if user_input:
                self.userInput.setDisabled(True)
                if config_manager.get_setting('chatGPTApiNoOfChoices') == 1:
                    self.sendButton.setText(config_manager.get_translation('stop'))
                    self.busyLoading = True
                    self.listView.setDisabled(True)
                    self.newButton.setDisabled(True)
                messages = self.get_messages(user_input)
                self.print(f'>>> {user_input}')
                self.save_data()
                self.currentLoadingID = self.contentID
                self.currentLoadingContent = self.contentView.toPlainText().strip()
                self.progressBar.show()  # show progress bar
                ChatGPTResponse(self).work_on_get_response(messages)  # get chatGPT response in a separate thread

    def file_names_without_extension(self, _dir: str, ext: str) -> None:
        files = glob.glob(os.path.join(_dir, f'*.{ext}'))
        return sorted([file[len(_dir) + 1 : -(len(ext) + 1)] for file in files if os.path.isfile(file)])

    def exec_python_file(self, script) -> None:
        if config_manager.get_setting('developer'):
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
        # The following config values can be modified with plugins, to extend functionalities
        config_manager.update_internal('predefinedContexts', '"[none]": "", "[custom]": ""')
        config_manager.update_internal('inputSuggestions', [])
        config_manager.update_internal('chatGPTTransformers', [])
        config_manager.update_internal('chatGPTApiFunctionSignatures', [])
        config_manager.update_internal('chatGPTApiAvailableFunctions', {})

        plugin_folder = Path.cwd() / 'plugins'
        # always run 'integrate google searches'
        internet_searches = 'integrate google searches'
        script = plugin_folder / f'{internet_searches}.py'
        self.exec_python_file(script)
        for plugin in self.file_names_without_extension(plugin_folder, 'py'):
            if plugin != internet_searches and plugin not in config_manager.get_setting('chatGPTPluginExcludeList'):
                script = plugin_folder / f'{plugin}.py'
                self.exec_python_file(script)
        # if internetSeraches in config_manager.get_setting("chatGPTPluginExcludeList"):
        #     config_manager.update_setting("chatGPTApiFunctionSignatures[0], ''")

    # FIXME: transformers
    def process_response(self, responses: str) -> None:
        if responses:
            # reload the working content in case users change it during waiting for response
            self.contentID = self.currentLoadingID
            self.contentView.setPlainText(self.currentLoadingContent)
            self.currentLoadingID = self.currentLoadingContent = ''
            # transform responses
            # for t in config_manager.get_setting('chatGPTTransformers'):
            #     responses = t(responses)
            # update new reponses
            self.print(responses)
            # scroll to the bottom
            if config_manager.get_setting('chatGPTApiAutoScrolling'):
                content_scroll_bar = self.contentView.verticalScrollBar()
                content_scroll_bar.setValue(content_scroll_bar.maximum())
            # if not (responses.startswith("OpenAI API re") or responses.startswith("Failed to connect to OpenAI API:")) and config_manager.get_setting("chatGPTApiAudio"):
            #        self.play_audio(responses)
        # empty user input
        self.userInput.setText('')
        # auto-save
        self.save_data()
        # hide progress bar
        self.userInput.setEnabled(True)
        if config_manager.get_setting('chatGPTApiNoOfChoices') == 1:
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
                gTTS(text=text, lang=config_manager.get_setting('chatGPTApiAudioLanguage') or 'en').save(
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

        # new_action = QAction(config_manager.get_translation('toggleDarkTheme'), self)
        # new_action.triggered.connect(self.toggle_theme)
        # customise_menu.addAction(new_action)

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
        for index, context in enumerate(config_manager.get_setting('predefinedContexts')):
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
        if plugin in config_manager.get_setting('chatGPTPluginExcludeList'):
            config_manager.get_internal('chatGPTPluginExcludeList').remove(plugin)
        else:
            config_manager.get_internal('chatGPTPluginExcludeList').append(plugin)
        internet_searches = 'integrate google searches'
        if internet_searches in config_manager.get_setting('chatGPTPluginExcludeList') and config_manager.get_setting('loadingInternetSearches') == 'auto':
            config_manager.update_setting('loadingInternetSearches', 'none')
        elif internet_searches not in config_manager.get_setting('chatGPTPluginExcludeList') and config_manager.get_setting('loadingInternetSearches') == 'none':
            config_manager.update_setting('loadingInternetSearches', 'auto')
            config_manager.update_setting('chatGPTApiFunctionCall', 'auto')
        # reload plugins
        config_manager.get_internal('chatGPTApi').runPlugins()

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
        database_directory = Path(config_manager.get_setting('chatGPTApiLastChatDatabase')).parent
        open_command = self.get_open_command()
        os.system(f'{open_command} {database_directory}')

    def open_plugins_directory(self) -> None:
        open_command = self.get_open_command()
        os.system(f'{open_command} plugins')

    def toggle_regexp(self) -> None:
        config_manager.update_setting('regexpSearchEnabled', str(not bool(config_manager.get_setting('regexpSearchEnabled'))))
        self.chatGPT.update_search_tool_tips()
        QMessageBox.information(self, 'ChatGPTQT', f"Regex for search and replace is {'enabled' if config_manager.get_setting('regexpSearchEnabled') else 'disabled'}!")

    def toggle_system_tray(self) -> None:
        config_manager.update_setting('enableSystemTray', not config_manager.get_setting('enableSystemTray'))
        QMessageBox.information(self, 'ChatGPTQT', 'You need to restart this application to make the changes effective.')

    def toggle_theme(self) -> None:
        config_manager.update_internal('darkTheme', not config_manager.get_internal('darkTheme'))
        # app = QApplication.instance()
        # if config_manager.get_setting('darkTheme'):
        #     styles.dark(app)
        # else:
        #     styles.light(app)
        # # Re-apply the style to the ModernWindow wrapper if necessary
        # mw = config_manager.get_internal('mainWindow')
        # if mw and isinstance(mw, windows.ModernWindow):
        #     # Ensure app style is set first, then apply to ModernWindow
        #     # This step might be redundant if ModernWindow automatically picks up app style changes.
        #     # However, explicitly setting it ensures the theme is applied.
        #     if config_manager.get_setting('darkTheme'):
        #         styles.dark(app)
        #     else:
        #         styles.light(app)
        #     mw.setStyleSheet(app.styleSheet())
        pass # Theme toggling disabled for now


    # Work with system tray
    def is_wayland(self) -> bool:
        return bool(platform.system() == 'Linux' and os.getenv('QT_QPA_PLATFORM') is not None and os.getenv('QT_QPA_PLATFORM') == 'wayland')

    def bring_to_foreground(self, window: QMainWindow) -> None: # window is now ModernWindow
        if window and not (window.isVisible() and window.isActiveWindow()):
            window.raise_()
            # Method activateWindow() does not work with qt.qpa.wayland
            # platform.system() == "Linux" and not os.getenv('QT_QPA_PLATFORM') is None and os.getenv('QT_QPA_PLATFORM') == "wayland"
            # The error message is received when QT_QPA_PLATFORM=wayland:
            # qt.qpa.wayland: Wayland does not support QWindow::requestActivate()
            # Therefore, we use hide and show methods instead with wayland.
            if window.isVisible() and not window.isActiveWindow():
                window.hide() # Should be called on ModernWindow instance
            window.show() # Should be called on ModernWindow instance
            if not self.is_wayland():
                window.activateWindow() # Should be called on ModernWindow instance


if __name__ == '__main__':

    def show_main_window() -> None:
        app = QApplication.instance() # Get app instance
        if not app: # Create if it doesn't exist (should be created before this function by `app = QApplication(sys.argv)`)
            app = QApplication(sys.argv)
            # config_manager.update_internal('app', app) # Not strictly necessary to store in config_manager here

        # Apply the dark theme QSS globally
        app.setStyleSheet(DARK_THEME_QSS)

        if not config_manager.get_internal('mainWindow'):
            main_window_instance = MainWindow()
            config_manager.update_internal('mainWindow', main_window_instance)
            main_window_instance.show()
        else:
            current_window = config_manager.get_internal('mainWindow')
            current_window.bring_to_foreground(current_window)


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
    if config_manager.get_setting('enableSystemTray'):
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
