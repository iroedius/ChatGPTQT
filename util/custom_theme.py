def get_dark_theme_qss():
    return """
QMainWindow {
    background-color: #2E2E2E;
    color: #E0E0E0;
}

QPushButton {
    background-color: #5699D6;
    color: white;
    border-radius: 4px;
    padding: 6px;
    border: 1px solid #4A89C8;
}
QPushButton:hover {
    background-color: #6BADF0;
}
QPushButton:pressed {
    background-color: #4A89C8;
}

QLineEdit {
    background-color: #3C3C3C;
    color: #E0E0E0;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px;
}
QLineEdit:focus {
    border: 1px solid #5699D6;
}

QPlainTextEdit {
    background-color: #3C3C3C;
    color: #E0E0E0;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px;
}
QPlainTextEdit:focus {
    border: 1px solid #5699D6;
}

QListView {
    background-color: #3C3C3C;
    color: #E0E0E0;
    border: 1px solid #555555;
    border-radius: 4px;
}
QListView::item:hover {
    background-color: #4A4A4A;
}
QListView::item:selected {
    background-color: #5699D6;
    color: white;
}

QComboBox {
    background-color: #3C3C3C;
    color: #E0E0E0;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px;
}
QComboBox:hover {
    background-color: #4A4A4A;
}
QComboBox:selected {
    background-color: #5699D6;
    color: white;
}
QComboBox QAbstractItemView { /* Dropdown list */
    background-color: #3C3C3C;
    color: #E0E0E0;
    border: 1px solid #555555;
    selection-background-color: #5699D6;
}

QCheckBox {
    color: #E0E0E0;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #555555;
    border-radius: 3px;
}
QCheckBox::indicator:unchecked {
    background-color: #3C3C3C;
}
QCheckBox::indicator:unchecked:hover {
    border: 1px solid #6BADF0;
}
QCheckBox::indicator:checked {
    background-color: #5699D6;
    border: 1px solid #4A89C8;
}
QCheckBox::indicator:checked:hover {
    background-color: #6BADF0;
    border: 1px solid #5699D6;
}

QLabel {
    color: #E0E0E0;
}

QMenuBar {
    background-color: #383838;
    color: #E0E0E0;
}
QMenuBar::item {
    background-color: transparent;
    padding: 4px 8px;
}
QMenuBar::item:selected {
    background-color: #5699D6;
    color: white;
}
QMenuBar::item:pressed {
    background-color: #4A89C8;
}

QMenu {
    background-color: #3C3C3C;
    color: #E0E0E0;
    border: 1px solid #555555;
}
QMenu::item {
    padding: 4px 20px 4px 20px;
}
QMenu::item:selected {
    background-color: #5699D6;
    color: white;
}
QMenu::separator {
    height: 1px;
    background-color: #555555;
    margin-left: 10px;
    margin-right: 5px;
}

QDialog {
    background-color: #2E2E2E;
    color: #E0E0E0;
}

QSplitter::handle {
    background-color: #383838;
    border: 1px solid #555555;
    width: 1px; /* or height for horizontal splitter */
    margin: 2px;
}
QSplitter::handle:hover {
    background-color: #5699D6;
}
QSplitter::handle:pressed {
    background-color: #4A89C8;
}

QProgressBar {
    border: 1px solid #555555;
    border-radius: 4px;
    text-align: center;
    color: #E0E0E0;
    background-color: #3C3C3C;
}
QProgressBar::chunk {
    background-color: #5699D6;
    width: 10px; /* Or some other appropriate value */
    margin: 0.5px;
}

/* QSystemTrayIcon is often styled by the system, but we can try */
/* This might not have a visible effect on all platforms */
QSystemTrayIcon {
    /* No specific QSS properties, as it's mostly system-dependent */
}
"""
