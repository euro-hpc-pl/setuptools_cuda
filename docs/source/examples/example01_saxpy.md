(sec:example01)=
# Example 01: CUDA Saxpy

## Overview

In this example we will discuss a simple project implementing the saxpy function run on 
CUDA enabled device. If you need a quick recap, saxpy is a function accepting a floating point 
number $a$ and two vectors $X, Y$ of the same length. It then performs the following operation:

$$
\require{mathtools}
Y \coloneqq aX + Y,
$$
(i.e. $Y$ is updated in-place). We will be writing our own kernel for saxpy, but to keep things 
simple we will not try to compute optimal size of the execution grid. Hence, our implementation 
will also accept parameters designating the number of threads in a thread block and the number 
of thread blocks. We want our function to work on numpy arrays, which means that we will have to 
handle copying data to and from the host. Having all that in mind, the function that we want to 
create in this extension has the following signature (we intentionally skip type hints, because, 
as you will soon see, we will use Cython for its definition):


```python
def saxpy(a, x, y, num_threads, num_blocks):
    ...
```

The full code for this example can be found 
[here](https://github.com/euro-hpc-pl/setuptools_cuda/tree/master/examples/01_basic), 
and below is a detailed explanation of each file that the example comprises.

## File structure

The file directory tree for our example looks as follows:

```text
├── pyproject.toml
├── saxpycu
│   ├── saxpy_impl.cu
│   ├── saxpy_impl.h
│   └── saxpy.pyx
├── setup.py
└── test
    └── test_saxpy.py
```

Briefly, the files play the following role:

- `saxpycu` directory contains source code of the extension:
   - `saxpy_impl.cu` contain the kernel code for saxpy, as well as a wrapper that allows running 
     it for host arrays.
   - `saxpy_impl.h` is a header file containing declaration of the wrapper. This is the file that 
     we'll include from the Cython source file.
   - `saxpy.pyx` is a Cython file containing the actual definition of the function that we will 
     call from Python.
- `pyproject.toml` defines metadata for the package that we want to bu ild, but most importantly 
  it defines **build dependencies**. This is where you declare your dependency on `setuptools-cuda`.
- `setup.py` is where you define the extension modules (it's currently not possible to do it in 
  `pyproject.toml`)
- `test/text_saxpy.py` is a file containing tests to gives us a reasonable confidence that what 
  we created actually works.

## Source of the extension

:::{note}
We are intentionally leaving our the error checking to not clutter the example code. However, in 
your production code you should *always* care about error checking. For instance, it might be 
wise to check if memory allocation on the device succeeded, or that the kernel was successfully 
launched.
:::

We will start by describing the actual content of the extension - the implementation of `saxpy`. 
The `saxpycu/saxpy_impl.cu` contains both a kernel and a wrapper that allow for running it from 
the host:

:::{literalinclude} ../../../examples/01_basic/saxpycu/saxpy_impl.cu
:::

First, we have the definition of the `_saxpy` kernel. As you can see, the kernel is templated, 
and can work on arrays of arbitrary (numerical) type. If you are familiar with CUDA programming, 
you shouldn't have any problems working out what the kernel does.

Next we have `saxpy_wrapper` function, which serves as a "launcher" of the `_saxpy` kernel. 
Importantly, using CUDA terminology, this is a host function. We would like to call it from our 
Cython code.

:::{note}
When writing wrappers such as `saxpy_wrapper`, make sure you always free the resources. 
Sometimes, you might get away with forgetting e.g. some call to `cudaFree`, but probably it will 
resurface later as a hard to track, low-level bug. If you are curious how it may look like, 
comment out one of `cudaFree` calls in `saxpy_wrapper`, rebuild the extension and launch the tests.
:::

Lastly, we have explicit template instantiations of the `saxpy_wrapper` for two values of 
parameter `T`, i.e. this code:

```cpp
template void saxpy_wrapper(float, float*, float*, int, int, int);
template void saxpy_wrapper(double, double*, double*, int, int, int);
```

Explicit instantiations tell the compiler to compile those variants of `saxpy_wrapper` even if 
they are not called anywhere in the code. **This is a very important detail**, and we'll explain 
its importance shortly.

## Header file

The header file is short, and contains the declaration of `saxpy_wrapper`. We will use 
it in our Cython (.pyx) file to declare `saxpy_wrapper` as external function.

:::{literalinclude} ../../../examples/01_basic/saxpycu/saxpy_impl.h
:::

## Cython file

In this example, Cython serves as a glue between C++ and Python. Of course, you don't have to 
use it in your projects, but for many use cases it simplifies things by a lot, compared to e.g. 
using NumPy's C API. Our Cython file looks as follows:

:::{literalinclude} ../../../examples/01_basic/saxpycu/saxpy.pyx
:::

We start by declaring that we are using C++. The default language is C, which won't let us 
compile our templates. Next, we define fused type `real`. In pure Python we would call `real`a 
*union* of `float` and `double`. If a function contains arguments of type real, Cython will 
prepare bot a version for both single and double precision floating-point numbers.

Next, we declare usage of the external function `saxpy_wrapper` defined in `saxpy_impl.h`. 
Compare this declaration:
```cython
cdef extern from "saxpy_impl.h":
    void saxpy_wrapper[T](T a, T * x, T * y, int n, int numThreads, int numBlocks);
```
with the definition we have in `saxpy_impl.h`:
```cpp
template <typename T>
void saxpy_wrapper(T a, T* x, T* y, int n, int numThreads, int numBlocks);
```
As you can see, the Cython version is just a simple rewrite of the C++ function into cythonic 
language.

Finally, we reach the main goal of this example, and define `saxpy` function. Aside from being 
Python function, it contains several differences as compared to the `saxpy_wrapper`:

- We use our fused type `real` instead of an abstract template argument `T`. We hence restrict 
  possible values of `T` with which `saxpy_wrapper` can be called to `float` and `double`. 
  Notice however, that we do this only to exemplify how the fused types work, in principle 
  nothing prevents you from using saxpy with e.g. values of type `int`. 

  Also, here is where the explicit instantiation of `saxpy_wrapper` comes into play. By using 
  `real` type, we make sure that the only variants of `saxpy_wrapper` that will be called by our 
  extension are ones with `T=float` and `T=double`. By explicitly instantiating those variants, 
  we make sure they are compiled and will be available at runtime. Without the explicit 
  instantiation, the compilation  would go smoothly, but at runtime we wil get low-level errors 
  telling us about the undefined symbols.
- Instead of using pointers, we use `real[::1]` syntax. Basically it means that we can accept a 
  continuous memory view, like e.g. a numpy array. To obtain the underlying pointer, we simply 
  extract address of the zero-th element (e.g. `&x[0]`).

  :::{note}
  Because we use `real[::1]` syntax as the type for our arrays, the `saxpy` function won't work for 
  numpy views. If we would like to allow views as well, we would have to do some additional work 
  which we will not discuss here.
  :::
- The `saxpy` function does not accept the explicit `n` argument designating the length of `x` 
  and `y`. This is simply because it's not needed, as we can just extract len of the `x` or `y` 
  array.

## pyproject.toml

The `pyproject.toml` of our package is simple and looks as follows:

:::{literalinclude} ../../../examples/01_basic/pyproject.toml
:::

In the `[build-system]` section we define the build-time requirements. We have 
`setuptoools-cuda` and `setuptools` for obvious reasons. Next we have wheel, which will allow us 
installing and building the package by simply running `pip install`. Lastly, we have `cython` as 
we will also use it for building our project.

IN the `[project]` section, we define the name of our package and its version, but also 
dependencies that will be installed when running `pip install`. We include `numpy`, because we 
will test our functions on arrays and also `pytest` for easily running our tests.

:::{note}
Typically, requirements needed for tests are placed in separate "extra" dependencies. However, 
we decided to put them directly in `dependencies` to make the example simpler.
:::

## setup.py

The `setup.py` is where we define our extension. It looks as follows;

:::{literalinclude} ../../../examples/01_basic/setup.py
:::

Let's break it down. As usually, we import the `setup` function from `setuptools`. We also 
import the `cythonize` function, because our extension contains `Cython` modules. Finally, we 
import `CudaExtension` from `setuptools_cuda`, which we will use to define our extension.

The extension is defined by creating and instance of `CudaExtension` class. It accepts the same 
arguments as regular extensions, in this example we used only the mandatory ones which is `name` 
and `sources`. Contrary to regular extensions, however, `CudaExtensions` need to be passed to 
`cuda_extensions` keyword of the `setup` function. Before doing so, we let the Cython do its 
magic and pass the extensions through Cythonize.

## Installing the package

Now that we know everything about our package, it is time to install it (and the extension it 
provides). Before you do this, we highly recommend that you define the `CUDAHOME` environmental 
variable. It should point to your CUDA installation location. For instance, you can do it by 
running the following command **in the same shell session** that you are going to use for 
installing the package (of course, adjust the actual path to your particular setup):
```shell
export CUDAHOME=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda
```
Refer to our [manual](sec:manual) for more thorough explanation of `CUDAHOME` variable.


To install the package, simply run the following command from the `examples/01_basic` directory. 
```shell
pip install .
```
If everything worked fine, the installation should proceed without errors.

## Running the tests

To run the tests, you should have working installation of Nvidia drivers and CUDA runtime. If 
this is the case, the tests can be run by simply launching the `pytest` command from the 
`examples/01_basic` directory. If you haven't modified the example yourself, the tests should 
pass without any errors.

If you are curious, the tests are simple and comprise a single test, run for multiple 
pseudo-randomly generated cases.

:::{literalinclude} ../../../examples/01_basic/test/test_saxpy.py
:::
