@echo off
setlocal
cd /d "%~dp0"
call ".\qc_env\Scripts\activate.bat"
python -m streamlit run "streamlit_nav_app3.py" --server.headless true
endlocal
