#include <sparseDataStruct/matrix_sparse.hpp>

__host__ MatrixSparse::MatrixSparse(int i_size, int j_size, int n_elements, MatrixType type, bool isDevice)
    : n_elements(n_elements), i_size(i_size), j_size(j_size), isDevice(isDevice), type(type)
{
    MemAlloc();
}

__host__ MatrixSparse::MatrixSparse(const MatrixSparse &m, bool copyToOtherMem)
    : MatrixSparse(m.i_size, m.j_size, m.n_elements, m.type, m.isDevice ^ copyToOtherMem)
{
    cudaMemcpyKind memCpy = (m.isDevice)
                                ? (isDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost
                                : (isDevice) ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost;
    gpuErrchk(cudaMemcpy(vals, m.vals, sizeof(T) * n_elements, memCpy));
    gpuErrchk(cudaMemcpy(colPtr, m.colPtr, sizeof(int) * n_elements, memCpy));
    gpuErrchk(cudaMemcpy(rowPtr, m.rowPtr, sizeof(int) * (type == CSR) ? i_size + 1 : n_elements, memCpy));
}

__host__ void MatrixSparse::MemAlloc()
{
    int rowPtrSize = (type == CSR) ? i_size + 1 : n_elements;
    if (isDevice)
    {
        gpuErrchk(cudaMalloc(&vals, n_elements * sizeof(T)));
        gpuErrchk(cudaMalloc(&rowPtr, rowPtrSize * sizeof(int)));
        gpuErrchk(cudaMalloc(&colPtr, n_elements * sizeof(int)));

        gpuErrchk(cudaMalloc(&_device, sizeof(MatrixSparse)));
        gpuErrchk(cudaMemcpy(_device, this, sizeof(MatrixSparse), cudaMemcpyHostToDevice));
    }
    else
    {
        vals = new T[n_elements];
        rowPtr = new int[rowPtrSize];
        colPtr = new int[n_elements];
    }
}

__host__ __device__ void MatrixSparse::AddElement(int k, int i, int j, const T val)
{
    vals[k] = val;
    colPtr[k] = j;
    if (type == CSR)
    {
        if (rowPtr[i + 1] == 0)
            rowPtr[i + 1] = rowPtr[i];
        rowPtr[i + 1]++;
    }
    else
    {
        rowPtr[k] = i;
    }
}

typedef cusparseStatus_t (*Func)(...);
__host__ void MatrixSparse::OperationCuSparse(void *operation, cusparseHandle_t &handle)
{
    ((Func)operation)(handle, i_size, j_size, n_elements, rowPtr, colPtr, vals);
}

__host__ void MatrixSparse::OperationCuSparse(void *operation, cusparseHandle_t &handle, size_t *pBufferSizeInBytes)
{
    ((Func)operation)(handle, i_size, j_size, n_elements, rowPtr, colPtr, pBufferSizeInBytes);
}

__host__ void MatrixSparse::OperationCuSparse(void *operation, cusparseHandle_t &handle, void *pBuffer)
{
    ((Func)operation)(handle, i_size, j_size, n_elements, rowPtr, colPtr, vals, pBuffer);
}

__host__ MatrixSparse::~MatrixSparse()
{
    if (isDevice)
    {
        gpuErrchk(cudaFree(vals));
        gpuErrchk(cudaFree(rowPtr));
        gpuErrchk(cudaFree(colPtr));
        gpuErrchk(cudaFree(_device));
    }
    else
    {
        delete[] vals;
        delete[] rowPtr;
        delete[] colPtr;
    }
}

__host__ __device__ void MatrixSparse::Print()
{
    for (int k = 0; k < n_elements; k++)
    {
        Get(k).Print();
    }
}