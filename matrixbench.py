""" Written by Benjamin Jack Cullen """
import sys
import time
import numpy as np
import cupy as cp


def cpupy(n1, n2):
    start_time = time.time()
    np_array = np.random.rand(n1, n2)
    np_result = np.matmul(np_array, np_array)
    print(f'\nnumpy time (CPU): {time.time() - start_time} seconds')
    print(f'np.size(np_result): {np.size(np_result)}')


def gpupy(n1, n2):
    start_time = time.time()
    cp_array = cp.random.rand(n1, n2)
    cp_result = cp.matmul(cp_array, cp_array)
    print(f'\ncupy (GPU): {time.time() - start_time} seconds')
    print(f'cp.size(cp_result): {cp.size(cp_result)}\n')


if '-h' in sys.argv:
    print('')
    print('[MATRIX BENCH]')
    print('                     CPU vs NVIDIA CUDA C GPU in Matrix Multiplication(s) Test.')
    print('')
    print('-n1     Size         Default 10000')
    print('-n2     Size         Default 10000')
    print('-cpu    Bench CPU    Optional (Can be used with -cpu)')
    print('-gpu    Bench GPU    Optional (Can be used with -gpu)')
    print('')
    print('[NOTE]  Number of matrix multiplications can be calculated by n1(n2).')
    print('        Example: n1(n2) = 100000(100000) = 100 million.')
    print('        Example operation: matrixbench.exe -n1 10000 -n2 10000 -cpu -gpu')
    print('        WARNING: Do NOT exceed n(n) = 30000(30000) = -n1 30000 -n2 30000 = 900 million matrix multiplications!')
    print('')
else:
    """ WARNING: Do not exceed 1 billion matrix multiplications!
    Matrix multiplication with NumPy (n, n2 = 10000, 10000 = 100 million matrix multiplications)
    Matrix multiplication with NumPy (n, n2 = 20000, 20000 = 400 million matrix multiplications)
    Matrix multiplication with NumPy (n, n2 = 30000, 30000 = 900 million matrix multiplications)
    """
    n1, n2 = 10000, 10000
    if '-n1' in sys.argv:
        n1 = int(sys.argv[sys.argv.index('-n1')+1])
    if '-n2' in sys.argv:
        n2 = int(sys.argv[sys.argv.index('-n2') + 1])

    if '-cpu' in sys.argv:
        cpupy(n1=n1, n2=n2)
    if '-gpu' in sys.argv:
        gpupy(n1=n1, n2=n2)
