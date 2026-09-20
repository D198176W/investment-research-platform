@echo off
chcp 65001 >nul
echo ================================
echo 智能投研平台 - 启动脚本
echo ================================
echo.

REM 检查 .env 文件
if not exist ".env" (
    echo [提示] 未找到 .env 文件，正在从 .env.example 复制...
    copy .env.example .env
    echo [警告] 请编辑 .env 文件并填入实际的 API 密钥！
    echo.
)

echo [1/2] 启动 FastAPI 后端服务...
cd backend\gateway
start "FastAPI Gateway" cmd /k "python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
cd ..\..

timeout /t 3 /nobreak >nul

echo [2/2] 启动 Streamlit 前端服务...
cd frontend
start "Streamlit Frontend" cmd /k "streamlit run streamlit_app.py --server.port 8501"
cd ..

echo.
echo ================================
echo 服务启动完成！
echo ================================
echo 后端 API: http://localhost:8000
echo API 文档: http://localhost:8000/docs
echo 前端界面: http://localhost:8501
echo ================================
echo.
pause
