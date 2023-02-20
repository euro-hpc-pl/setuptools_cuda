# Examples

Since a code snippet is more than a thousand words, we believe that the best 
way to learn `setuptools-cuda` is to look at the examples. If you are 
feeling adventurous, you can go and view them directly on our GitHub. 
However, we recommend reading at least the description of example 01 below, 
as it will introduce you to all the concepts needed for writing your own 
extensions.

## Prerequisites

Both examples require you to have the following:

- Working Python >= 3.9 installation
- Ability to install Python packages. We strongly recommend you create a 
  fresh virtual environment for the purpose of experimenting with 
  `setuptools-cuda`.
- Working installation of CUDA toolkit.
- (optionally but recommended) Working installation of NVidia driver and 
  CUDA runtime (otherwise you will be able to build but not run the examples).

The `setuptools-cuda` uses a `CUDAHOME` environmental variable, which you 
should set to the path of your CUDA installation. If you don't do this, 
`setuptools-cuda` will try to auto-detect it, but it will probably fail and 
you will have a bad time trying to figure out why things aren't working.

## Currently available examples

### [Example 01: CUDA Saxpy](sec:example01)

This example shows how the basic saxpy implementation works with 
`setuptools-cuda`.

### [Example 02: Integration with Thrust library](sec:example02)

The second example demonstrates how `setuptools-cuda` can be integrated with 
thrust. 

:::{toctree}
:hidden:

example01_saxpy.md
example02_thrust.md
:::
