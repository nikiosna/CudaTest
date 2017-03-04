# CudaTest
A simple programm based on JCuda to measure the time for adding n numbers via CUDA.

Currently only works on linux (tested on Linux Mint  18.1)

Installation:
  1. Download zip from https://github.com/nikiosna/CudaTest/releases
  2. extract and "cd CudaTest"
  3. (float  test) java -jar CudaTest.jar -f 
     (double test) java -jar CudaTest.jar -d

Syntax:
    necessary
    -d (--double) OR -f (--float)
    additional
    -n (--number) [int]   Change the default value from 10 000 000 to a chosen number

Building:
    TODO
    
Libraries
    JCuda https://github.com/jcuda/jcuda-main