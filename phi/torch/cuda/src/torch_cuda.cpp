#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cstdio>

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h> // SpMM, SpMV
#include <vector>
#include "torch_cuda.hpp"

#define TRUE 1
#define FALSE 0

void *globalBuffer = NULL;
size_t globalBufferSize = 0;
cusparseHandle_t globalHandle = NULL;
cusparseSpMatDescr_t globalSparseMatrixA;
torch::Tensor dC;
cusparseDnVecDescr_t dC_cusparse;

void allocateBuffer(size_t newSize) {
    if(newSize > globalBufferSize) {
        CHECK_CUDA( cudaFree(globalBuffer) )
        CHECK_CUDA( cudaMalloc(&globalBuffer, newSize) )
        globalBufferSize = newSize;
    }
}

torch::Tensor cusparse_SpMM(
                        const at::Tensor& dA_csrOffsets,
                        const at::Tensor& dA_columns,
                        const at::Tensor& dA_values,
                        const at::Tensor& dB,
                        const int dim_i,
                        const int dim_j)
{
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    cusparseHandle_t handle = NULL;
    void* dBuffer         = NULL;
    size_t bufferSize     = 0;
    int dim_k             = dB.size(1);
    auto A_nnz          = dA_values.size(0);
    float alpha           = 1.0f;
    float beta            = 0.0f;

    torch::Tensor dC = torch::empty({dim_i, dim_k}, dB.options());
    CHECK_CUSPARSE( cusparseCreate(&handle) )

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, dim_i, dim_j, A_nnz,
                                      dA_csrOffsets.data_ptr<int>(), dA_columns.data_ptr<int>(),
                                      dA_values.data_ptr<float>(),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, dim_j, dim_k, dim_k, dB.data_ptr<float>(),
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, dim_i, dim_k, dim_k, dC.data_ptr<float>(),
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG2, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG2, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    return dC;
}

torch::Tensor cusparse_SpMV(
                        at::Tensor& dB,
                        const int dim_i,
                        const int dim_j)
{

    cusparseDnVecDescr_t vecX;
    size_t               bufferSize = 0;
    float alpha           = 1.0f;
    float beta            = 0.0f;
    dB = dB.view({dim_j});

    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, dim_j, dB.data_ptr<float>(), CUDA_R_32F) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 globalHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, globalSparseMatrixA, vecX, &beta, dC_cusparse, CUDA_R_32F,
                                 CUSPARSE_SPMV_CSR_ALG2, &bufferSize) )
    allocateBuffer(bufferSize);

   // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(globalHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, globalSparseMatrixA, vecX, &beta, dC_cusparse, CUDA_R_32F,
                                 CUSPARSE_SPMV_CSR_ALG2, globalBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    dB = dB.view({1, dim_j});
    dC = dC.view({1, dim_i});
    return dC;
}


torch::Tensor cusparse_SpMM_BA(
                        const at::Tensor& dA_csrOffsets,
                        const at::Tensor& dA_columns,
                        const at::Tensor& dA_values,
                        const at::Tensor& dB,
                        const int dim_j,
                        const int dim_k)
{
    /*
    cudaEvent_t startSolver;
    cudaEvent_t stopSolver;
    cudaEventCreate( &startSolver );
    cudaEventCreate( &stopSolver );
    cudaEventRecord( startSolver, 0);   // <<<<< start capturing time
    */
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    cusparseHandle_t handle = NULL;
    void* dBuffer         = NULL;
    size_t bufferSize     = 0;
    int dim_i             = dB.size(0);
    auto   A_nnz          = dA_values.size(0);
    float alpha           = 1.0f;
    float beta            = 0.0f;

    torch::Tensor dC = torch::empty({dim_k, dim_i}, dB.options());
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    int version;
    cusparseGetVersion(handle, &version);
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, dim_j, dim_k, A_nnz,
                                      dA_csrOffsets.data_ptr<int>(), dA_columns.data_ptr<int>(),
                                      dA_values.data_ptr<float>(),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, dim_j, dim_i, dim_j, dB.data_ptr<float>(),
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, dim_k, dim_i, dim_k, dC.data_ptr<float>(),
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG1, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG1, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CHECK_CUDA( cudaFree(dBuffer) )
    dC = dC.view({dim_i, dim_k});
    /*
    cudaEventRecord( stopSolver, 0 );   // <<<<< stop capturing time
    cudaEventSynchronize( stopSolver );
    cudaEventElapsedTime( &elapsedTime, startSolver, stopSolver );
    std::cout << "el: " << elapsedTime << std::endl;*/
    return dC;
}

template<typename T>
torch::Tensor create_gpu_tensor(std::vector<T> data, at::IntArrayRef shape) {
    if(typeid(T) == typeid(float)){
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto tens = torch::from_blob(&data[0], shape, options);
        auto gpu_tens = tens.to(torch::Device(torch::kCUDA, 0));
        return gpu_tens;
    }
    else if(typeid(T) == typeid(int)) {
        auto options = torch::TensorOptions().dtype(torch::kInt);
        auto tens = torch::from_blob(&data[0], shape, options);
        auto gpu_tens = tens.to(torch::Device(torch::kCUDA, 0));
        return gpu_tens;
    }
    else {
        fprintf(stderr, "error: %s: create_gpu_tensor type undexpected at line %d\n", __FILE__,       \
                __LINE__);                                                     \
        exit(1);
    }

}

torch::Tensor nan_division(torch::Tensor a, torch::Tensor b) {
    auto res = a/b;
    res.index_put_({~torch::isfinite(res)}, 0);
    return res;
}

template<typename T>
void print_variable(char* name, T variable) {
    std::cout << name << "\n" << variable << std::endl;
}

std::vector<std::vector<torch::Tensor>> conjugate_gradient(torch::Tensor csr_values, torch::Tensor csr_cols, torch::Tensor csr_rows, int csr_dim0, int csr_dim1, int nnz,
                        torch::Tensor y, torch::Tensor x,  float rtol, float atol, torch::Tensor max_iter, bool trj) {
/* ASSUMPTIONS
        x0 is a matrix that contains columns of vectors (each vector is an independent problem to solve)
        y is a matrix that contains columns of vectors (each vector is an independent problem to solve)
*/
    // CREATE HANDLE AND CSR MATRIX REPRESENTATION
    CHECK_CUSPARSE( cusparseCreate(&globalHandle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&globalSparseMatrixA, csr_dim0, csr_dim1, nnz,
                                      csr_rows.data_ptr<int>(), csr_cols.data_ptr<int>(),
                                      csr_values.data_ptr<float>(),
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    dC = torch::empty({csr_dim0}, x.options());
    CHECK_CUSPARSE( cusparseCreateDnVec(&dC_cusparse, csr_dim0, dC.data_ptr<float>(), CUDA_R_32F) )
    // CREATE HANDLE AND CSR MATRIX REPRESENTATION

    std::vector<std::vector<torch::Tensor>> trajectory;
    torch::Tensor sum_y = torch::sum(torch::mul(y,y), -1);
    torch::Tensor atol_sq = create_gpu_tensor(std::vector<float>{atol*atol}, torch::IntArrayRef{1});
    torch::Tensor tolerance_sq = torch::maximum(rtol * rtol * sum_y, atol_sq); // max([ , , ...], [atol*atol])
    torch::Tensor lin_x = cusparse_SpMV(x, csr_dim0, csr_dim1);
    auto residual = y - lin_x;
    auto dx = residual;
    int it_counter = 0;
    auto residual_squared = torch::sum(torch::mul(residual, residual), -1, TRUE);
    auto rsq0 = residual_squared;
    auto diverged = torch::any(~x.isfinite(), -1, TRUE);
    auto iterations = torch::zeros_like(diverged, x.options());
    auto function_evaluations = torch::ones_like(iterations, x.options());
    auto converged = torch::all(residual_squared <= tolerance_sq, -1, TRUE);
    auto finished = converged | diverged | (iterations >= max_iter);
    auto not_finished = ~finished;
    auto dummy_ones = torch::ones_like(finished, torch::kCUDA);
    while(!torch::equal(finished, dummy_ones)) {
        it_counter += 1;
        iterations = iterations + not_finished;
        torch::Tensor dy = cusparse_SpMV(dx, csr_dim0, csr_dim1);
        function_evaluations += not_finished;
        auto dx_dy = torch::sum(residual * dy, -1, TRUE);
        auto step_size = nan_division(residual_squared, dx_dy); // Account for nan values
        step_size = torch::mul(step_size, not_finished);
        x += (step_size * dx);
        if(it_counter % 50 == 0) {
            residual = y - cusparse_SpMV(x, csr_dim0, csr_dim1);
            function_evaluations += 1;
        }
        else {
            residual = (residual - step_size * dy);
        }
        auto residual_squared_old = residual_squared;
        residual_squared = torch::sum(torch::mul(residual, residual), -1, TRUE);
        dx = residual + (nan_division(residual_squared, residual_squared_old)) * dx; // Account for nan values
        diverged = torch::any(residual_squared / rsq0 > 100) & (iterations >= 8);
        converged = torch::all(residual_squared <= tolerance_sq, -1, TRUE);
        finished = converged | diverged | (iterations >= max_iter);
        not_finished = ~finished;
        if(trj) {
            trajectory.push_back({x, residual, torch::squeeze(iterations, -1), torch::squeeze(function_evaluations, -1), torch::squeeze(converged, -1),
            torch::squeeze(diverged, -1)});
        }
    }
    if(!trj){
        trajectory.push_back({x, residual, torch::squeeze(iterations, -1), torch::squeeze(function_evaluations, -1), torch::squeeze(converged, -1),
            torch::squeeze(diverged, -1)});
    }

    // CLOSE HANDLE AND CSR MATRIX REPRESENTATION
    CHECK_CUSPARSE( cusparseDestroySpMat(globalSparseMatrixA) )
    CHECK_CUSPARSE( cusparseDestroy(globalHandle) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(dC_cusparse) )

    //CHECK_CUDA( cudaFree(globalBuffer) )
    // CLOSE HANDLE AND CSR MATRIX REPRESENTATION
    return trajectory;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conjugate_gradient", &conjugate_gradient, "Conjugate gradient function");
  m.def("cusparse_SpMM", &cusparse_SpMM, "Sparse(CSR) times dense matrix multiplication on CUSPARSE");
  m.def("cusparse_SpMV", &cusparse_SpMV, "Dense vector times Sparse(CSR) matrix multiplication on CUSPARSE");
}