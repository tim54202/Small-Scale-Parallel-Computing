#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
        fprintf(stderr, "Unable to open file %s\n", filename);
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

double* generateRandomVector(int size) {
    double *vector = (double *)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        vector[i] = (double)(rand() % 10); // Random number
    }
    return vector;
}

// Sparse matrix and vector multiplication
void csrMatrixVectorMultiply(CSRMatrix csrMatrix, double *vector, double *result) {
#pragma omp parallel for
    for (int i = 0; i < csrMatrix.M; i++) {
        result[i] = 0;
        for (int j = csrMatrix.IRP[i]; j < csrMatrix.IRP[i + 1]; j++) {
            result[i] += csrMatrix.AS[j] * vector[csrMatrix.JA[j]];
        }
    }
}

int main() {
    srand(time(NULL)); // Initialize random seed

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
    int maxThreads = omp_get_max_threads();

    for (int fileIdx = 0; fileIdx < numFiles; fileIdx++) {
        const char* filename = filenames[fileIdx];
        CSRMatrix csrMatrix = readMatrixMarketToCSR(filename);

        for (int kIndex = 0; kIndex < numKValues; kIndex++) {
            int k = kValues[kIndex];

            for (int t = 1; t <= maxThreads; t++) {
                omp_set_num_threads(t);
                double totalExecTime = 0.0;
                int numRuns = 8; // Number of runs per thread configuration

                for (int run = 0; run < numRuns; run++) {
                    double *vector = generateRandomVector(csrMatrix.N * k);
                    double *result = (double *)malloc(csrMatrix.M * k * sizeof(double));

                    double start_time = omp_get_wtime();
                    for (int col = 0; col < k; col++) {
                        csrMatrixVectorMultiply(csrMatrix, vector + col * csrMatrix.N, result + col * csrMatrix.M);
                    }
                    double end_time = omp_get_wtime();

                    totalExecTime += end_time - start_time;

                    free(vector);
                    free(result);
                }

                double avgExecTime = totalExecTime / numRuns;
                double flops = (2.0 * csrMatrix.NZ * k) / avgExecTime;
                if (t == maxThreads) { // Print only at max thread count
                    printf("File: %s, Threads: %d, k: %d, Average Execution Time: %lf seconds, FLOPS: %lf\n",
                           filename, t, k, avgExecTime, flops);
                }
            }
        }

        freeCSR(csrMatrix);
    }

    return 0;
}
