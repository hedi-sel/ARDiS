cd $(dirname $0)

rm dist/*
python setup.py bdist_wheel
pip uninstall -y dist/*.whl
pip install dist/*.whl

cd ../