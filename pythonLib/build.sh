cd $(dirname $0)

rm dist/*
python3.8 setup.py bdist_wheel
python3.8 -m pip uninstall -y dist/*.whl
python3.8 -m pip install dist/*.whl

cd ../