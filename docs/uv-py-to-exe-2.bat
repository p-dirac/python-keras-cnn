setLocal

set p="src/cnnpy"
set libs=".venv\Lib\site-packages"

set h1="pkg_resources"
set h2="pkg_resources.extern"

--copy-metadata scikeras

set b1=".venv\Lib\site-packages\numpy.libs:."
set b2=".venv\Lib\site-packages\pandas.libs:."

uvx pyinstaller --onefile --copy-metadata scikeras --hidden-import=%h1% --hidden-import=%h2% --paths=%p%;%libs% --add-binary %b1% --add-binary %b2% %p%/app.py


endLocal