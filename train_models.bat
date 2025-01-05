@echo off
echo [%date% %time%] ========================================
echo [%date% %time%] Starting Crypto Model Training Pipeline
echo [%date% %time%] ========================================
echo.

echo [%date% %time%] Activating Virtual Environment...
call C:\DataLoaders\AiTrainer\venv\Scripts\activate.bat
echo [%date% %time%] Virtual Environment Activated: %VIRTUAL_ENV%
echo.

cd /d "C:\DataLoaders\AiTrainer"
echo [%date% %time%] Changed to working directory: %CD%
echo.

echo [%date% %time%] Starting Model Training...
python crypto_model_trainer.py
echo [%date% %time%] Model Training Complete
echo.

echo [%date% %time%] Starting Performance Analysis...
python analyze_social_impact.py
echo [%date% %time%] Performance Analysis Complete
echo.

echo [%date% %time%] Deactivating Virtual Environment...
deactivate
echo [%date% %time%] Virtual Environment Deactivated
echo.

echo [%date% %time%] ========================================
echo [%date% %time%] Crypto Model Training Pipeline Complete
echo [%date% %time%] ======================================== 