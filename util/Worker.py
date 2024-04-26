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
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self) -> None:
        """
        Initialise the runner function with passed args, kwargs.
        """

        # assign a reference to this current thread
        # config_manager.get_setting('workerThread') = QThread.currentThread()

        # Retrieve args/kwargs here; and fire processing using them
        try:
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
        insert_string = 'import config\nconfig_manager.get_setting("pythonFunctionResponse") = '
        code = re.sub('^!(.*?)$', r'import os\nos.system(\1)', code, flags=re.MULTILINE)
        if '\n' in code:
            substrings = code.rsplit('\n', 1)
            last_line = re.sub(r'print\((.*)\)', r'\1', substrings[-1])
            code = code if last_line.startswith(' ') else f'{substrings[0]}\n{insert_string}{last_line}'
        else:
            code = f'{insert_string}{code}'
        return code

    def get_function_reponse(self, response_message: dict, function_name: str) -> str:
        if function_name == 'python':
            config_manager.update_setting('pythonFunctionResponse', '')
            python_code = textwrap.dedent(response_message['function_call']['arguments'])
            refined_code = self.fine_tune_python_code(python_code)

            print('--------------------')
            print('running python code ...')
            if config_manager.get_setting('developer') or config.codeDisplay:
                print('```')
                print(python_code)
                print('```')
            print('--------------------')

            try:
                exec(refined_code, globals())
                function_response = str(config_manager.get_setting('pythonFunctionResponse'))
            except Exception:
                function_response = python_code
            info = {'information': function_response}
            function_response = json.dumps(info)
        else:
            fuction_to_call = config_manager.get_setting('chatGPTApiAvailableFunctions')[function_name]
            function_args = json.loads(response_message['function_call']['arguments'])
            function_response = fuction_to_call(function_args)
        return function_response

    def get_stream_dunction_response_message(self, completion, function_name: str) -> dict:
        function_arguments = ''
        for event in completion:
            delta = event['choices'][0]['delta']
            if delta and delta.get('function_call'):
                function_arguments += delta['function_call']['arguments']
        return {
            'role': 'assistant',
            'content': None,
            'function_call': {
                'name': function_name,
                'arguments': function_arguments,
            },
        }

    def run_completion(self, this_message: list, progress_callback):
        self.functionJustCalled = False

        def runThisCompletion(thisThisMessage):
            if config_manager.get_setting('chatGPTApiFunctionSignatures') and not self.functionJustCalled:
                return openai.ChatCompletion.create(
                    model=config_manager.get_setting('chatGPTApiModel'),
                    messages=thisThisMessage,
                    n=1,
                    temperature=config_manager.get_setting('chatGPTApiTemperature'),
                    max_tokens=config_manager.get_setting('chatGPTApiMaxTokens'),
                    functions=config_manager.get_setting('chatGPTApiFunctionSignatures'),
                    function_call=config_manager.get_setting('chatGPTApiFunctionCall'),
                    stream=True,
                )
            return openai.ChatCompletion.create(
                model=config_manager.get_setting('chatGPTApiModel'),
                messages=thisThisMessage,
                n=1,
                temperature=config_manager.get_setting('chatGPTApiTemperature'),
                max_tokens=config_manager.get_setting('chatGPTApiMaxTokens'),
                stream=True,
            )

        while True:
            completion = runThisCompletion(this_message)
            function_name = ''
            try:
                # consume the first delta
                for event in completion:
                    delta = event['choices'][0]['delta']
                    # Check if a function is called
                    if not delta.get('function_call'):
                        self.functionJustCalled = True
                    elif 'name' in delta['function_call']:
                        function_name = delta['function_call']['name']
                    # check the first delta is enough
                    break
                # Continue only when a function is called
                if self.functionJustCalled:
                    break

                # get stream function response message
                response_message = self.get_stream_dunction_response_message(completion, function_name)

                # get function response
                function_response = self.get_function_reponse(response_message, function_name)

                # process function response
                # send the info on the function call and function response to GPT
                this_message.append(response_message)  # extend conversation with assistant's reply
                this_message.append(
                    {
                        'role': 'function',
                        'name': function_name,
                        'content': function_response,
                    },
                )  # extend conversation with function response

                self.functionJustCalled = True

                if not config_manager.get_setting('chatAfterFunctionCalled'):
                    progress_callback.emit('\n\n~~~ ')
                    progress_callback.emit(function_response)
                    return None
            except Exception:
                logging.exception('Exception while getting ChatCompletion')
                self.show_errors()
                break

        return completion

    def show_errors(self) -> None:
        if config_manager.get_setting('developer'):
            print(traceback.format_exc())

    def get_response(self, messages, progress_callback, functionJustCalled: bool = False):
        responses = ''
        if config_manager.get_setting('loadingInternetSearches') == 'always' and not functionJustCalled:
            # print("loading internet searches ...")
            try:
                completion = openai.ChatCompletion.create(
                    model=config_manager.get_setting('chatGPTApiModel'),
                    messages=messages,
                    max_tokens=config_manager.get_setting('chatGPTApiMaxTokens'),
                    temperature=config_manager.get_setting('chatGPTApiTemperature'),
                    n=1,
                    functions=config_manager.get_setting('integrate_google_searches_signature'),
                    function_call={'name': 'integrate_google_searches'},
                )
                response_message = completion['choices'][0]['message']
                if response_message.get('function_call'):
                    function_args = json.loads(response_message['function_call']['arguments'])
                    fuction_to_call = config_manager.get_setting('chatGPTApiAvailableFunctions').get('integrate_google_searches')
                    function_response = fuction_to_call(function_args)
                    messages.append(response_message)  # extend conversation with assistant's reply
                    messages.append(
                        {
                            'role': 'function',
                            'name': 'integrate_google_searches',
                            'content': function_response,
                        },
                    )
            except Exception:
                logging.exception('Unable to load internet resources.')
        try:
            if config_manager.get_setting('chatGPTApiNoOfChoices') == 1:
                completion = self.run_completion(messages, progress_callback)
                if completion is not None:
                    progress_callback.emit('\n\n~~~ ')
                    for event in completion:
                        # stop generating response
                        stop_file = Path('.stop_chatgpt')
                        if stop_file.is_file():
                            stop_file.unlink()
                            break
                        # RETRIEVE THE TEXT FROM THE RESPONSE
                        event_text = event['choices'][0]['delta']  # EVENT DELTA RESPONSE
                        progress = event_text.get('content', '')  # RETRIEVE CONTENT
                        # STREAM THE ANSWER
                        progress_callback.emit(progress)
            else:
                if config_manager.get_setting('chatGPTApiFunctionSignatures'):
                    completion = openai.ChatCompletion.create(
                        model=config_manager.get_setting('chatGPTApiModel'),
                        messages=messages,
                        max_tokens=config_manager.get_setting('chatGPTApiMaxTokens'),
                        temperature=0.0 if config_manager.get_setting('chatGPTApiPredefinedContext') == 'Execute Python Code' else config.chatGPTApiTemperature,
                        n=config_manager.get_setting('chatGPTApiNoOfChoices'),
                        functions=config_manager.get_setting('chatGPTApiFunctionSignatures'),
                        function_call={'name': 'run_python'} if config_manager.get_setting('chatGPTApiPredefinedContext') == 'Execute Python Code'
                        else config_manager.get_setting('chatGPTApiFunctionCall'),
                    )
                else:
                    completion = openai.ChatCompletion.create(
                        model=config_manager.get_setting('chatGPTApiModel'),
                        messages=messages,
                        max_tokens=config_manager.get_setting('chatGPTApiMaxTokens'),
                        temperature=config_manager.get_setting('chatGPTApiTemperature'),
                        n=config_manager.get_setting('chatGPTApiNoOfChoices'),
                    )

                response_message = completion['choices'][0]['message']
                if response_message.get('function_call'):
                    function_name = response_message['function_call']['name']
                    if function_name == 'python':
                        config_manager.update_setting('pythonFunctionResponse', '')
                        function_args = response_message['function_call']['arguments']
                        insert_string = 'import config\nconfig_manager.get_setting("pythonFunctionResponse") = '
                        if '\n' in function_args:
                            substrings = function_args.rsplit('\n', 1)
                            new_function_args = f'{substrings[0]}\n{insert_string}{substrings[-1]}'
                        else:
                            new_function_args = f'{insert_string}{function_args}'
                        try:
                            exec(new_function_args, globals())
                            function_response = str(config_manager.get_setting('pythonFunctionResponse'))
                        except Exception:
                            function_response = function_args
                        info = {'information': function_response}
                        function_response = json.dumps(info)
                    else:
                        # if not function_name in config_manager.get_setting('chatGPTApiAvailableFunctions'):
                        #    print("unexpected function name: ", function_name)
                        fuction_to_call = config_manager.get_setting('chatGPTApiAvailableFunctions').get(function_name, 'integrate_google_searches')
                        try:
                            function_args = json.loads(response_message['function_call']['arguments'])
                        except Exception:
                            function_args = response_message['function_call']['arguments']
                            if function_name == 'integrate_google_searches':
                                function_args = {'keywords': function_args}
                        function_response = fuction_to_call(function_args)

                    # check function response
                    # print("Got this function response:", function_response)

                    # process function response
                    # send the info on the function call and function response to GPT
                    messages.append(response_message)  # extend conversation with assistant's reply
                    messages.append(
                        {
                            'role': 'function',
                            'name': function_name,
                            'content': function_response,
                        },
                    )  # extend conversation with function response
                    if config_manager.get_setting('chatAfterFunctionCalled'):
                        return self.get_response(messages, progress_callback, functionJustCalled=True)
                    responses += f'{function_response}\n\n'

                for index, choice in enumerate(completion.choices):
                    chat_response = choice.message.content
                    if chat_response:
                        if len(completion.choices) > 1:
                            if index > 0:
                                responses += '\n'
                            responses += f'~~~ Response {(index + 1)}:\n'
                        responses += f'{chat_response}\n\n'
        # error codes: https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        except openai.error.APIError as e:
            # Handle API error here, e.g. retry or log
            return f'OpenAI API returned an API Error: {e}'
        except openai.error.APIConnectionError as e:
            # Handle connection error here
            return f'Failed to connect to OpenAI API: {e}'
        except openai.error.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            return f'OpenAI API request exceeded rate limit: {e}'
        except Exception:
            # traceback.print_exc()
            responses = traceback.format_exc()
        return responses

    def work_on_get_response(self, messages) -> None:
        # Pass the function to execute
        worker = Worker(self.get_response, messages)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.parent.process_response)
        worker.signals.progress.connect(self.parent.print_stream)
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
