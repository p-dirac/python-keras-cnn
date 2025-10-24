setLocal

set p="src/cnnpy"
set libs=".venv\Lib\site-packages"


uvx pyinstaller --additional-hooks-dir=%p% --paths=%p%;%libs% --onefile  %p%/app.py


endLocal