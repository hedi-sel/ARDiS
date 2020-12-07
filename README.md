
Requirements
====================

  * CMake (3.10+), g++-7 (C++ 11 standard)
  * CUDA Compatible Graphics card, and CUDA toolkit (10.2+).
  * Python 3.6+, with the following modules: 
    * Numpy 
    * Matplotlib

<h3> Optional </h3>

  * WolframScript (1.3.0+)
    * (for building your own reactor shape)

Building
====================

Make directories 'build' and 'output'

Unpack data.rar

Build library:
```c++
cd build && cmake ../
make
```
Install library with pip:
```c++
cd pythonLib
python setup.py bdist_wheel
pip install dist/*.whl

```

Run Test
====================

```c++
python example/MinimumExample.py 
```