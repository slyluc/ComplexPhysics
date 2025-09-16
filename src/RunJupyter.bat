@echo off
:: Go to your repo folder
cd /d "C:\Users\lucq0\OneDrive\Skrivebord\Complex Physics\src"

:: Launch Jupyter with large buffer size
jupyter notebook --NotebookApp.max_buffer_size=32000000000
