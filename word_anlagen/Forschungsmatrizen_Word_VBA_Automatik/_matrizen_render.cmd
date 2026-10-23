@echo off
cd /d "D:\OLD\Sync\Projekte\Masterarbeit\Project\PoC4\word_anlagen\Forschungsmatrizen_Word_VBA_Automatik"
py -3 "D:\OLD\Sync\Projekte\Masterarbeit\Project\PoC4\word_anlagen\Forschungsmatrizen_Word_VBA_Automatik\forschungsmatrizen.py" --config "D:\OLD\Sync\Projekte\Masterarbeit\Project\PoC4\word_anlagen\Forschungsmatrizen_Word_VBA_Automatik\matrizen.json" --ausgabe "D:\OLD\Sync\Projekte\Masterarbeit\Project\PoC4\word_anlagen\Forschungsmatrizen_Word_VBA_Automatik" > "D:\OLD\Sync\Projekte\Masterarbeit\Project\PoC4\word_anlagen\Forschungsmatrizen_Word_VBA_Automatik\render_protokoll.txt" 2>&1
exit /b %ERRORLEVEL%
