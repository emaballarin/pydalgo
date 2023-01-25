#!/usr/bin/env bash
NPPATH="$(python3 -c 'import numpy; print(numpy.__path__[0])')/core/include/numpy"
mkdir -p .lib/python
PRE_PYTHONPATH="$PYTHONPATH"
export PYTHONPATH="$PYTHONPATH:./lib/python/"
CFLAGS="-I$NPPATH" python3 setup.py install --home=./
python3 setup.py build
GIBBSC=$(find build/ -name "_gibbs*.so")
cp $GIBBSC ./dimension/
export PYTHONPATH="$PRE_PYTHONPATH"
python -m pip install ./
