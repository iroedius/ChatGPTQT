# ChatGPT-GUI

## ChatGPT Graphical User Interface

A Qt-based graphical user interface application for accessing Large Language Models. It supports models from OpenAI (ChatGPT) and Google (Gemini).

Repository: https://github.com/eliranwong/ChatGPT-GUI

Developer: Eliran Wong

<img width="1243" alt="screenshot" src="https://user-images.githubusercontent.com/25262722/227805265-bd26c0c9-9c6a-4e4d-83c9-3e27ea3d3c7e.png">

# Background

We integrated ChatGPT in one of our gui applications, [UniqueBible.app](https://github.com/eliranwong/UniqueBible/wiki/Bible-Chat-with-ChatGPT-API).  Here in this project, we modify the codes to make ChatGPT-GUI as a standalone application for wider purposes, now including support for multiple LLM providers.

# Cross-platform

Winodws, macOS, Linux, ChromeOS are supported.

You may also run on Android via Termux.

## LLM API Configuration

This application supports Large Language Models (LLMs) from multiple providers. You will need to configure the respective API keys for the provider you wish to use.

### Supported LLM Providers

Currently, ChatGPT-GUI supports the following LLM providers:

- **OpenAI**: For accessing models like GPT-3.5, GPT-4, etc.
- **Google Gemini**: For accessing Google's Gemini family of models.

### Selecting the LLM Provider

You can choose your active LLM provider in the application's settings:
1. Click the "Settings" button (usually found at the bottom-right of the chat interface).
2. In the settings dialog, locate the "LLM Provider" dropdown menu.
3. Select either "openai" or "gemini" from the dropdown.
4. Configure the API key and other relevant settings for your selected provider (see details below for each provider). The fields for the non-selected provider may become hidden or disabled.
5. Click "OK" to save the settings. The application will then use the selected provider for generating responses, and the main window title will update to reflect the current provider.

### OpenAI Configuration

- **OpenAI API Key**: Your unique API key from OpenAI. This is required if using OpenAI.
  - Generate API key at: https://platform.openai.com/account/api-keys
- **Organization ID (Optional)**: If your API key is part of an organization, you may need to enter this.
- **OpenAI API Model**: Select the specific OpenAI model you wish to use (e.g., `gpt-3.5-turbo`, `gpt-4-turbo-preview`). Ensure your API key has access to the selected model.
  - For ChatGPT-4 Users: Make sure you have access to 'gpt-4' models and enter both the API Key and Organization ID.

Read OpenAI pricing at: https://openai.com/pricing

### Google Gemini Configuration

- **Gemini API Key**: Your API key for the Gemini API. This is required if using Gemini.
  - You'll typically need to enable the "Generative Language API" (sometimes referred to as "Vertex AI Gemini API") in your Google Cloud project and create an API key associated with it.
- **Gemini Model Name**: The name of the Gemini model you wish to use (e.g., `gemini-pro`, `gemini-1.5-pro-latest`). The default is `gemini-pro`.

### Relevant `settings.json` Keys

While configuration is primarily done through the UI, these are the main keys stored in `settings.json`:
- `selected_llm_provider`: Stores the currently active provider, either "openai" or "gemini".
- `openai_api_key`: Stores the OpenAI API key.
- `openai_api_organization`: Stores the OpenAI organization ID.
- `chat_gpt_api_model`: Stores the selected OpenAI model name. (Note: this key name is from when only ChatGPT was supported).
- `gemini_api_key`: Stores the Gemini API key.
- `gemini_model_name`: Stores the selected Gemini model name.

# Difference between "ChatGPT-GUI" interface and ChatGPT web version

ChatGPT web version is available at: https://chat.openai.com/chat

"ChatGPT-GUI" uses the same model (when OpenAI is selected), but with enhanced features not available at ChatGPT web version.

With "ChatGPT-GUI", users can:

* include latest internet search results in ChatGPT responses (OpenAI-specific feature via function calling)

* enter multiline-message

* predefine context for conversations.  With "ChatGPT-GUI", users can specify a context for conversations.  For example, enter "talk about English literature" as the chat context in "Chat Settings", to get ChatGPT responses related to "English literature".  In addition, users can can choose to apply their customised contexts only in the beginning of a chat or all inputs.

* adjust temperature [What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.]

* adjust number of choices in ChatGPT responses [How many chat completion choices to generate for each input message.] (Currently OpenAI-specific)

* adjust font size for text display

* use python plugins, to automate tasks, add predefined context, or to process ChatGPT responses before they are displayed (some plugin features like function calling are currently OpenAI-specific)

* edit, print and save conversations

* save conversations for offline use

* search history, based on title or content

* perform search and replace on chat content

* oranize chat history into different separate database files

* enter message with voice-typing

* use OpenAI image model to generate images (OpenAI-specific)

* support system tray

* choose to use regular expression for search and replace

# Setup

For Windows users:

> https://github.com/eliranwong/ChatGPT-GUI/wiki/Setup-%E2%80%90-Windows

For macOS, Linux, ChromeOS users

> https://github.com/eliranwong/ChatGPT-GUI/wiki/Setup-%E2%80%90-macOS,-Linux,-ChromeOS

# QuickStart

1) Launch ChatGPT-GUI by running "python3 ChatGPT-GUI.py"
2) Click the "Settings" button to select your LLM Provider (OpenAI or Gemini) and enter the corresponding API key.
3) Enter your message and click the "Send" button, to start a conversation.

# User Interface

Graphical User Interface

> https://github.com/eliranwong/ChatGPT-GUI/wiki/UI-%E2%80%90-Graphical-User-Interface

Chat Settings (API Settings Dialog)

> https://github.com/eliranwong/ChatGPT-GUI/wiki/UI-%E2%80%90-Chat-Settings (Note: This wiki page may need updates to reflect multi-provider settings)

# Include Latest Internet Search Results

https://github.com/eliranwong/ChatGPT-GUI/wiki/Include-Latest-Internet-Search-Results (Currently an OpenAI-specific feature)

# Plugins

ChatGPT-GUI supports plugins, written in python, to extend functionalities. Some advanced plugin features like function calling are currently specific to the OpenAI provider.

How to use python plugins to process ChatGPT responses before they are displayed?

https://github.com/eliranwong/ChatGPT-GUI/wiki/Plugins-%E2%80%90-Transform-ChatGPT-Responses

How to use plugins to customize input suggestion?

> https://github.com/eliranwong/ChatGPT-GUI/wiki/Plugins-%E2%80%90-Input-Suggestions

How to use plugins to customize predefined contexts?

> https://github.com/eliranwong/ChatGPT-GUI/wiki/Plugins-%E2%80%90-Predefined-Contexts

How to use ChatGPT function calling features with plugins?

> https://github.com/eliranwong/ChatGPT-GUI/wiki/Plugins-%E2%80%90-ChatGPT-Function-Calling (OpenAI-specific)

# FAQ - Frequently Asked Questions

https://github.com/eliranwong/ChatGPT-GUI/wiki/FAQ-%E2%80%90-Frequently-Asked-Questions