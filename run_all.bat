@echo off
cd /d F:\Smart-Traffic-Violation-Detection-System

call .\.venv\Scripts\activate

start "FASTAPI" cmd /k python -m uvicorn app.api.main:app --reload
timeout /t 2 >nul
start "STREAMLIT" cmd /k python -m streamlit run app\ui\streamlit_app.py --server.port 8501
