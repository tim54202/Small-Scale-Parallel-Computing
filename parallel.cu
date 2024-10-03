#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

typedef struct {
    int M;
    int N;
    int NZ;
    int *IRP;
    int *JA;
    double *AS;
} CSRMatrix;

// Convert from MatrixMarket format to CSR format
CSRMatrix readMatrixMarketToCSR(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Can't open the file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char line[1024];
    // Skip file header
    while (!feof(file)) {
        fgets(line, sizeof(line), file);
        if (line[0] != '%') break;
    }

    int M, N, NZ;
    sscanf(line, "%d %d %d", &M, &N, &NZ);

    CSRMatrix csr;
    csr.M = M;
    csr.N = N;
    csr.NZ = NZ;
    csr.IRP = (int *)malloc((M + 1) * sizeof(int));
    csr.JA = (int *)malloc(NZ * sizeof(int));
    csr.AS = (double *)malloc(NZ * sizeof(double));

    int *rowCounts = (int *)calloc(M, sizeof(int));
    int row, col;
    double val;

    // First pass to count non-zero elements per row
    while (fscanf(file, "%d %d %lf", &row, &col, &val) != EOF) {
        rowCounts[row - 1]++;
    }

    // Build IRP array
    csr.IRP[0] = 0;
    for (int i = 1; i <= M; i++) {
        csr.IRP[i] = csr.IRP[i - 1] + rowCounts[i - 1];
    }

    fseek(file, 0, SEEK_SET); // Reset file pointer to beginning
    // Skip header
    while (!feof(file)) {
        fgets(line, sizeof(line), file);
        if (line[0] != '%') break;
    }

    // Second pass to populate JA and AS arrays
    int *currentRowCounts = (int *)calloc(M, sizeof(int));
    while (fscanf(file, "%d %d %lf", &row, &col, &val) != EOF) {
        int rowIndex = row - 1;
        int pos = csr.IRP[rowIndex] + currentRowCounts[rowIndex];
        csr.JA[pos] = col - 1; // 0-based indexing
        csr.AS[pos] = val;
        currentRowCounts[rowIndex]++;
    }

    free(rowCounts);
    free(currentRowCounts);
    fclose(file);

    return csr;
}

void freeCSR(CSRMatrix csr) {
    free(csr.IRP); // Free row pointer array
    free(csr.JA);  // Free column index array
    free(csr.AS);  // Free non-zero element value array
}

double* generateVector(int M, int k) {
    double *vector = (double *)malloc(M * k * sizeof(double));
    for (int i = 0; i < M * k; i++) {
        vector[i] = (double)(rand() % 10); // Random number
    }
    return vector;
}

__global__ void multiplyKernel(double *d_A, int *d_IRP, int *d_JA, double *d_X, double *d_Y, int M, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        for (int j = 0; j < k; j++) {
            double sum = 0;
            for (int idx = d_IRP[i]; idx < d_IRP[i + 1]; idx++) {
                sum += d_A[idx] * d_X[d_JA[idx] * k + j];
            }
            d_Y[i * k + j] = sum;
        }
    }
}

int main() {
    srand(time(NULL)); // Initialize random number seed

    const char* filenames[] = {
            "cage4.mtx", "mhda416.mtx", "mcfe.mtx", "olm1000.mtx",
            "adder_dcop_32.mtx", "west2021.mtx", "cavity10.mtx",
            "rdist2.mtx", "cant.mtx", "olafu.mtx", "Cube_Coup_dt0.mtx",
            "ML_Laplace.mtx", "bcsstk17.mtx", "mac_econ_fwd500.mtx",
            "mhd4800a.mtx", "cop20k_A.mtx", "raefsky2.mtx",
            "af23560.mtx", "lung2.mtx", "PR02R.mtx", "FEM_3D_thermal1.mtx",
            "thermal1.mtx", "thermal2.mtx", "thermomech_TK.mtx",
            "nlpkkt80.mtx", "webbase-1M.mtx", "dc1.mtx",
            "amazon0302.mtx", "af_1_k101.mtx", "roadNet-PA.mtx"
    };
    int numFiles = sizeof(filenames) / sizeof(filenames[0]);
    int kValues[] = {1, 2, 3, 6}; // Different values of k
    int numKValues = sizeof(kValues) / sizeof(kValues[0]);
    int threadCounts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int numThreadCounts = sizeof(threadCounts) / sizeof(threadCounts[0]);

    for (int fileIdx = 0; fileIdx < numFiles; fileIdx++) {
        const char* filename = filenames[fileIdx];
        CSRMatrix csrMatrix = readMatrixMarketToCSR(filename);

        double *d_A, *d_X, *d_Y;
        int *d_IRP, *d_JA;

        // Allocate GPU memory
        cudaMalloc((void **)&d_A, csrMatrix.NZ * sizeof(double));
        cudaMalloc((void **)&d_IRP, (csrMatrix.M + 1) * sizeof(int));
        cudaMalloc((void **)&d_JA, csrMatrix.NZ * sizeof(int));

        // Copy data to device
        cudaMemcpy(d_A, csrMatrix.AS, csrMatrix.NZ * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_IRP, csrMatrix.IRP, (csrMatrix.M + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_JA, csrMatrix.JA, csrMatrix.NZ * sizeof(int), cudaMemcpyHostToDevice);

        for (int kIndex = 0; kIndex < numKValues; kIndex++) {
            int k = kValues[kIndex];
            double *vector = generateVector(csrMatrix.N, k);
            double *result = (double *)malloc(csrMatrix.M * k * sizeof(double));


            // Allocate GPU memory for vector and result
            cudaMalloc((void **)&d_X, csrMatrix.N * k * sizeof(double));
            cudaMalloc((void **)&d_Y, csrMatrix.M * k * sizeof(double));
            cudaMemcpy(d_X, vector, csrMatrix.N * k * sizeof(double), cudaMemcpyHostToDevice);

            for (int threadIdx = 0; threadIdx < numThreadCounts; threadIdx++) {
                int numThreads = threadCounts[threadIdx];
                int numBlocks = (csrMatrix.M + numThreads - 1) / numThreads;
                double totalExecTime = 0.0;
                int numRuns = 8;

                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);

                for (int run = 0; run < numRuns; run++) {
                    cudaEventRecord(start);
                    multiplyKernel<<<numBlocks, numThreads>>>(d_A, d_IRP, d_JA, d_X, d_Y, csrMatrix.M, k);
                    cudaEventRecord(stop);

                    cudaEventSynchronize(stop);
                    float milliseconds = 0;
                    cudaEventElapsedTime(&milliseconds, start, stop);
                    totalExecTime += milliseconds / 1000.0; // Convert to seconds
                }

                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                double avgExecTime = totalExecTime / numRuns;
                double flops = (2.0 * csrMatrix.NZ * k) / avgExecTime;
                printf("File: %s, Threads: %d, k: %d, Average Execution Time: %lf seconds, FLOPS: %lf\n",
                       filename, numThreads, k, avgExecTime, flops);
            }

            cudaFree(d_X);
            cudaFree(d_Y);
            free(vector);
            free(result);
        }

        cudaFree(d_A);
        cudaFree(d_IRP);
        cudaFree(d_JA);
        freeCSR(csrMatrix);
    }

    return 0;
}