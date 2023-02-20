(sec:example02)=
# Example 02: Using Thrust library

## Overview

This example demonstrates how `setuptools_cuda` can be used in conjunction with Thrust library. 
We highly recommend that you first read the [saxpy example](sec:example01) description first, as 
here we will only focus on the important differences.

In this example, we will create a simple `sort` function that will sort a numpy array using 
`thrust::sort` on a CUDA enabled device. Internally, we will have to handle the transfer of data 
from numpy to the device, and then in the opposite way. Our signature will therefore look as 
follows

```python
def sort(data):
   ...
```
As previously, we will use Cython to make our life easier.

##  Directory structure

```text
├── pyproject.toml
├── setup.py
├── test
│   └── test_thrustcu.py
└── thrustcu
    ├── thrustcu_impl.cu
    ├── thrustcu_impl.h
    └── thrustcu.pyx

```
The directory structure in this example is pretty similar to the one in the first example, and 
hence we will not discuss the role of each individual file and instead we'll focus on the 
relevant contents.

## Main file of the extension

Compared to the previous example, this time we do not implement our own kernel. Instead, we rely 
on thrust to perform the heavy lifting. The source code for the main file of the extension looks 
as follows:

:::{literalinclude} ../../../examples/02_thrust/thrustcu/thrustcu_impl.cu
:language: c++
:::

We start by including several thrust files:

- `thrust/copy.h` for copying data between host and device.
- `thrust/device_vector.h` for definition of a device vector, which is a structure similar to a 
  `vector` in standard C++ library. 
- `thrust/sort.h` for the actual implementation of parallel sorting.

As previously, we use templating to allow usage of several data types in our function. As to the 
`_sort` function itself, it performs the following operations:

- Creates a `device_vector` `data_vec`. By passing it a range of pointers, we initialize the 
  device data to the contents of the original array.
- Sorts the created `device_vector`.
- Copies the data from the device vector back to the host array passed as the argument.

Similarly to the first example, we explicitly instantiate `_sort` template with several 
different data types. The fact that we used more data types then previously will be explored 
later on.

We also create a header file for the `_sort` function, which contains its declaration.

## Cython file

Our Cython file shares many similarities to the one from the first example. What's different is 
that we use numpy types in the `humber` fused type:

:::{literalinclude} ../../../examples/02_thrust/thrustcu/thrustcu.pyx
:language: cython
:::

Note that here we use `cimport` (it's not a typo, there's **c** there). The `cimport` 
instruction is used for importing stuff from another Cython module. If we used `import numpy` 
instead, all of those dtypes would be simply treated as Python objects and wouldn't work in a 
`fused` type declaration.

## Installing and running the tests

We once again remind you that before installing the package using `setuptools_cuda` you should 
define a `CUDAHOME` environmental variable pointing to your CUDA installation location.

Similarly to the first example, the package can be installed by running
```shell
pip install .
```
from the examples/02_thrust directory, and the tests can be launched by running:

```shell
pytest
```
from the same directory.
