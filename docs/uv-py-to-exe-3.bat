setLocal

set p="src/cnnpy"
set libs=".venv\Lib\site-packages"

set h1="importlib_metadata"

set b1=".venv\Lib\site-packages\numpy.libs:."
set b2=".venv\Lib\site-packages\pandas.libs:."

uvx pyinstaller --onefile --hidden-import=%h1%  --paths=%p%;%libs% --add-binary %b1% --add-binary %b2% %p%/app.py


endLocal