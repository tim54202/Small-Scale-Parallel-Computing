# Exploring Parallelism in Sparse Matrix Multiplication with OpenMP and CUDA
## Project Overview
This project explores the implementation of parallel programming techniques for sparse matrix multiplication using OpenMP and CUDA. Sparse matrices are prevalent in computational fields like scientific computing and data analysis, where efficient storage and processing are crucial. The project compares the performance of serial, OpenMP, and CUDA implementations of sparse matrix-vector multiplication, highlighting the computational benefits of parallelism.

## Key Technologies
OpenMP: A shared-memory parallel programming model used to execute multiple threads concurrently on multi-core CPUs.
CUDA: A parallel computing platform developed by NVIDIA that leverages the power of GPUs to perform large-scale computations efficiently.
CSR Format: The Compressed Sparse Row (CSR) format is used to store the sparse matrix for efficient memory utilization.
![Github](https://github.com/tim54202/High-Performance-Computing/blob/main/Images/CSR-Transformation.png)
## Methodology
CSR Format Conversion: The sparse matrix is first converted to CSR (Compressed Sparse Row) format to reduce memory usage. The Values, Col_Index, and Row_Ptr arrays are used to store non-zero elements and their positions.

Generating Random 2D Vector: A 2D vector with random double-precision floating-point numbers is generated to be multiplied by the sparse matrix.

OpenMP Implementation: The sparse matrix multiplication is implemented using OpenMP. The code parallelizes the for-loop iterations using #pragma omp parallel for, distributing the workload across multiple threads. The performance is evaluated by varying the number of threads from 1 to 16 to test scalability.

CUDA Implementation: The CUDA implementation leverages GPU parallelism by distributing the matrix-vector multiplication tasks across multiple GPU threads. The multiplyKernel function performs the matrix-vector multiplication, with different thread counts used to evaluate performance.

Performance Evaluation: The performance of both OpenMP and CUDA implementations is measured using FLOPS (Floating Point Operations Per Second). CUDA's performance is compared across various thread counts, while OpenMP's performance is analyzed by increasing the number of CPU threads.

## Results
FLOPS Analysis: The project analyzes the FLOPS performance of both OpenMP and CUDA implementations, with CUDA demonstrating significant advantages in parallel computing due to its ability to handle larger thread counts.

## Performance Comparison:

OpenMP: Shows good scalability with increasing thread counts, but reaches a plateau beyond 16 threads.
CUDA: Provides excellent performance gains for large-scale computations, though performance drops slightly when too many threads are used due to overhead.
## Conclusion
The project demonstrates that CUDA significantly outperforms OpenMP and the serial implementation in sparse matrix-vector multiplication, especially for larger matrices and higher computational loads. OpenMP is a suitable solution for moderate parallelism, while CUDA excels in high-performance scenarios with large datasets.
