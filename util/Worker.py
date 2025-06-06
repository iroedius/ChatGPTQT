import json
import logging
import re
import sys
import textwrap
import traceback
from collections.abc import Callable
from pathlib import Path

import openai
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot  # noreorder
from PySide6.QtWidgets import QWidget

from config_manager import ConfigManager

config_manager = ConfigManager()


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    """
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(str)


class Worker(QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    """

    def __init__(self, fn: Callable, *args, **kwargs) -> None:
        super().__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        # The 'progress_callback' might be used by fn if it's designed for streaming.
        # self.kwargs['progress_callback'] = self.signals.progress # Keep if fn expects it

    @Slot()
    def run(self) -> None:
        """
        Initialise the runner function with passed args, kwargs.
        """
        # Retrieve args/kwargs here; and fire processing using them
        try:
            # Check if progress_callback should be passed to self.fn
            # If self.fn is self.get_response, it now takes (user_prompt, chat_history, progress_callback)
            # The progress_callback is for streaming partial responses.
            # The LLMProvider.generate_response currently returns a full string, so true streaming
            # from provider through worker isn't fully set up yet.
            # For now, progress_callback in kwargs might not be used by the provider directly.
            if 'progress_callback' not in self.kwargs and hasattr(self.signals, 'progress'):
                 self.kwargs['progress_callback'] = self.signals.progress

            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            logging.exception('Exception in the "run" slot')
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class ChatGPTResponse:
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__()
        self.parent = parent
        self.threadpool = QThreadPool()

    def fine_tune_python_code(self, code: str) -> str:
        # This string will be exec'd. It needs to be self-contained.
        # It should instantiate ConfigManager and call update_setting.
        insert_string = ('from config_manager import ConfigManager\n'
                         'config_manager_instance = ConfigManager()\n'
                         'config_manager_instance.update_setting("python_function_response", ')
        code = re.sub('^!(.*?)$', r'import os\nos.system(\1)', code, flags=re.MULTILINE)
        if '\n' in code:
            substrings = code.rsplit('\n', 1)
            last_line = re.sub(r'print\((.*)\)', r'\1', substrings[-1])
            # Ensure the value part of update_setting is closed with a parenthesis
            code = code if last_line.startswith(' ') else f'{substrings[0]}\n{insert_string}{last_line})'
        else:
            code = f'{insert_string}{code})'
        return code

    # Removed get_function_reponse, get_stream_dunction_response_message, run_completion as they are
    # very specific to OpenAI direct calls and streaming/function calling features not yet
    # generalized in the LLMProvider abstraction.
    # The new get_response will directly use the llm_provider.

    def show_errors(self) -> None:
        if config_manager.get_setting('developer'):
            print(traceback.format_exc())

    def get_response(self, user_prompt: str, chat_history: list[dict], progress_callback: Signal):
        """
        Fetches response from the currently configured LLM provider.
        The progress_callback is available if we want to adapt providers to stream later.
        For now, providers return full string, so it will be emitted once.
        """
        try:
            # These settings are now primarily for the provider; provider loads them from config.
            # However, if we want to allow overriding them per call, they can be passed.
            # For now, let's rely on provider's loaded config, matching LLMProvider signature.
            temperature = self.parent.config_manager.get_setting(
                'chat_gpt_api_temperature' if self.parent.llm_provider.config_manager.get_setting('selected_llm_provider') == 'openai'
                else 'gemini_temperature', # Placeholder for potential Gemini-specific temp setting
                default=0.8 # A general default
            )
            max_tokens = self.parent.config_manager.get_setting(
                'chat_gpt_api_max_tokens' if self.parent.llm_provider.config_manager.get_setting('selected_llm_provider') == 'openai'
                else 'gemini_max_tokens', # Placeholder for potential Gemini-specific max_tokens
                default=1024 # A general default
            )

            # Function calling and multiple choices (n > 1) are more complex and provider-specific.
            # The current LLMProvider.generate_response is simplified and returns a single string.
            # Thus, the logic for chat_gpt_api_no_of_choices > 1 and function calling
            # from the original get_response is omitted here for simplification.
            # It would need to be built into the provider implementations or a more complex abstraction.

            response_text = self.parent.llm_provider.generate_response(
                prompt=user_prompt,
                chat_history=chat_history,
                temperature=temperature, # Pass along if desired, or let provider use its own config
                max_tokens=max_tokens    # Pass along if desired
            )

            # Since provider returns full text, emit it once via progress_callback
            # which is connected to self.parent.print_stream
            if response_text:
                 progress_callback.emit('\n\n~~~ ') # Separator used by print_stream
                 progress_callback.emit(response_text)
            return response_text # This will go to worker.signals.result -> self.parent.process_response

        except Exception as e:
            logging.exception("Exception in ChatGPTResponse.get_response (Worker)")
            error_message = f"Error during LLM call: {e}\n{traceback.format_exc()}"
            # Emit the error prefixed with the standard separator for assistant messages
            if progress_callback: # Check if progress_callback was successfully passed
                progress_callback.emit('\n\n~~~ ')
                progress_callback.emit(error_message)
            return error_message # Also return as result for process_response to handle

    def work_on_get_response(self, user_prompt: str, chat_history: list[dict]) -> None:
        # Pass the function to execute: self.get_response
        # Args for self.get_response: user_prompt, chat_history
        # The progress_callback is implicitly passed via self.kwargs['progress_callback'] in Worker.__init__
        worker = Worker(self.get_response, user_prompt, chat_history, progress_callback=self.signals.progress)
        worker.signals.result.connect(self.parent.process_response)
        worker.signals.progress.connect(self.parent.print_stream) # print_stream handles UI update
        # Connection
        # worker.signals.finished.connect(None)
        # Execute
        self.threadpool.start(worker)


class OpenAIImage:
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__()
        self.parent = parent
        self.threadpool = QThreadPool()

    def get_response(self, prompt: str, progress_callback: Callable | None = None):
        try:
            # https://platform.openai.com/docs/guides/images/introduction
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size='1024x1024',
            )
            return response['data'][0]['url']
        # error codes: https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        except openai.error.APIError as e:
            # Handle API error here, e.g. retry or log
            print(f'OpenAI API returned an API Error: {e}')
        except openai.error.APIConnectionError as e:
            # Handle connection error here
            print(f'Failed to connect to OpenAI API: {e}')
        except openai.error.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(f'OpenAI API request exceeded rate limit: {e}')
        except Exception:
            traceback.print_exc()
        return ''

    def work_on_get_response(self, prompt: str) -> None:
        # Pass the function to execute
        worker = Worker(self.get_response, prompt)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.parent.display_image)
        # Connection
        # worker.signals.finished.connect(None)
        # Execute
        self.threadpool.start(worker)
