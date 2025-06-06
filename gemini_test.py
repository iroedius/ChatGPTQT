try:
    import google.generativeai as genai
    print("Gemini library (google.generativeai) imported successfully!")
    if hasattr(genai, '__version__'):
        print(f"Version: {genai.__version__}")
    else:
        print("Version attribute not found, but import was successful.")
except ImportError as e:
    print(f"Error importing Gemini library: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
