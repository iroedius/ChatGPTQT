# 'config_manager' is now injected into the global scope when this plugin is run.

# Get the current input_suggestions list
suggestions = config_manager.get_setting('input_suggestions')

# Ensure it's a list before extending
if not isinstance(suggestions, list):
    suggestions = []

# Add new suggestions (avoiding duplicates if run multiple times, though not strictly necessary here)
new_items = ["Write a summary", "Write an outline", "Write a letter"]
for item in new_items:
    if item not in suggestions:
        suggestions.append(item)

# Save the updated list back to settings
config_manager.update_setting('input_suggestions', suggestions)