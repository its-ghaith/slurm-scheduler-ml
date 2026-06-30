@echo off
cd /d "%~dp0"
taskkill /f /im streamlit.exe 2>nul
call .venv\Scripts\activate.bat
streamlit run src/countcv/focidet/gui/foci_counter.py --server.address localhost
