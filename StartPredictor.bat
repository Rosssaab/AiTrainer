@echo off
cd /d "C:\DataLoaders\AiTrainer"
echo [%date% %time%] Activating Virtual Environment...
call C:\DataLoaders\AiTrainer\venv\Scripts\activate.bat
echo [%date% %time%] Virtual Environment Activated: %VIRTUAL_ENV%
echo..........................................
echo Starting Cyrpto_Predcitor.py- %date% %time%
python crypto_predictor.py
echo Predictions Complete - %date% %time% 