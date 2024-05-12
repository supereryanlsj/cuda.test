#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include "gemv.cuh"

void __global__ gemv(
  float* A, float* B, float* C,
  int outputRow, int outputCol, int hiddenDim) {
  // Warp index
  int h = blockIdx.x; // 0~B.x
  //int by = blockIdx.y; // 0~A.y/32

  // Thread index
  int hh = threadIdx.x / 32; // [0, 4)
  int sgId = threadIdx.x & 0x1f; // [0, 32)
  //int ty = threadIdx.y; // 0~1

  int offsetA = h * 32 * hiddenDim + hh * 32;
  int offsetB;
  int offsetC = h *  32;
  float aa[32];
  float bb[32];
  float cc;
  float ccSum[4];
  __shared__ float ccShm[32 * 4];

  offsetB = hh * 32;
#pragma unroll
  for (int nn = 0; nn < 32; nn++) {
    bb[nn] = B[offsetB + sgId];
    offsetB += 128;
  }

  for (int n = 0; n < 32; n++) {
    cc = 0;
#pragma unroll
    for (int nn = 0; nn < 32; nn++) {
      aa[nn] = A[offsetA + sgId];
      offsetA += 128;
    }

#pragma unroll
    for (int nn = 0; nn < 32; nn++) {
      cc += aa[nn] * bb[nn];
    }
#pragma unroll
    for (int l = 32; l > 1; l >>= 1) {
      cc += __shfl_xor_sync(0xffffffff, cc, l - 1, 32);
    }

    if (sgId == 0) {
      ccShm[n + 32 * hh] = cc;
    }
  }

  __syncthreads();

  if (hh == 0) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
      ccSum[k] = ccShm[32 * k + sgId];
    }

#pragma unroll
    for (int k = 1; k < 4; k++) {
      ccSum[0] += ccSum[k];
    }

    C[offsetC + sgId] = ccSum[0];
  }
}

void __global__ gemvQ40(
  uint8_t* A, float* B, float* C,
  int outputRow, int outputCol, int hiddenDim) {
  // Warp index
  int h = blockIdx.x; // 0~B.x
  //int by = blockIdx.y; // 0~A.y/32

  // Thread index
  int hh = threadIdx.x / 32; // [0, 4)
  int sgId = threadIdx.x & 0x1f; // [0, 32)
  //int ty = threadIdx.y; // 0~1

  int offsetA = h * 32 * hiddenDim / 8 + hh * 32;
  int offsetB;
  int offsetC = h * 32;
  uchar4 aaa[4];
  float aa[32];
  float bb[32];
  float cc;
  float ccSum[4];
  __shared__ float ccShm[32 * 4];

  offsetB = hh * 32;
  uchar4* aTemp = (uchar4*)A;
#pragma unroll
  for (int nn = 0; nn < 32; nn++) {
    bb[nn] = B[offsetB + sgId];
    offsetB += 128;
  }

  for (int n = 0; n < 32; n++) {
    cc = 0;
#pragma unroll
    for (int nn = 0; nn < 4; nn++) {
      aaa[nn] = aTemp[offsetA + sgId];
      offsetA += 128;
    }

#pragma unroll
    for (int nn = 0; nn < 16; nn++) {
      int nnWave = nn >> 2; // [0, 4)
      int nnIdx = nn & 0x3; // [0, 4)
      int shflLane = (sgId >> 2) + nnIdx * 8;
      int uchar4ArrayIdx = nnWave;
      int uchar4ArrayOffset = (sgId & 3);
      aa[2 * nn] = __shfl_sync(0xffffffff, aaa[uchar4ArrayIdx].x, shflLane, 32) & 0xf;
      aa[2 * nn + 1] = __shfl_sync(0xffffffff, aaa[uchar4ArrayIdx].y, shflLane, 32) >> 4;
      if (h == 0 && hh == 0 && n == 0) {
        printf("sgId: %d, nn = %d, shflLane = %d, uchar4ArrayIdx = %d, uchar4ArrayOffset = %d, aa[2 * nn] = %f, aa[2 * nn + 1] = %f\n", sgId, nn, shflLane, uchar4ArrayIdx, uchar4ArrayOffset, aa[2 * nn], aa[2 * nn + 1]);
      }
    }

#pragma unroll
    for (int nn = 0; nn < 32; nn++) {
      aa[nn] = aa[nn] - 8.0f;
    }

#pragma unroll
    for (int nn = 0; nn < 32; nn++) {
      cc += aa[nn] * bb[nn];
    }
#pragma unroll
    for (int l = 32; l > 1; l >>= 1) {
      cc += __shfl_xor_sync(0xffffffff, cc, l - 1, 32);
    }

    if (sgId == 0) {
      ccShm[n + 32 * hh] = cc;
    }
  }

  __syncthreads();

  if (hh == 0) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
      ccSum[k] = ccShm[32 * k + sgId];
    }

#pragma unroll
    for (int k = 1; k < 4; k++) {
      ccSum[0] += ccSum[k];
    }

    C[offsetC + sgId] = ccSum[0];
  }
}

int testGemvQ40(int m, int n, int k, int numbIter) {
  // Allocate host memory for matrices A and B
  cudaStream_t stream;
  cudaEvent_t start, stop;

  size_t sizeA = m * k / 2;
  sizeA = sizeA * 9 / 8;
  size_t sizeB = n * k * sizeof(float);
  size_t sizeC = m * n * sizeof(float);
  uint8_t* hostA;
  float* hostB;
  float* hostC;
  uint16_t* hostQuant;

  uint8_t* deviceA[10];
  uint8_t* deviceB[10];
  uint8_t* deviceC[10];
  cudaError_t ret;
  dim3 threads(128, 1);
  dim3 grid(m / 32, 1);
  float msecTotal = 0.0f;

  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  ret = cudaMallocHost(&hostA, sizeA);
  ret = cudaMallocHost(&hostB, sizeB);
  ret = cudaMallocHost(&hostC, sizeC);
  hostQuant = (uint16_t*)(hostA + m * k / 2);
  if (k > m) {
    for (int row = 0; row < m; row++) {
      int temp = (k - m + row) / 2;
      for (int col = 0; col < k / 2; col++) {
        if (col == temp) {
          if (temp & 0x1) {
            hostA[row * k / 2 + col] = 0x98;
          } else {
            hostA[row * k / 2 + col] = 0x89;
          }
        } else {
          hostA[row * k / 2 + col] = 0x88;
        }
      }
    }
  } else {
    for (int row = 0; row < m; row++) {
      int temp = row / 2;
      for (int col = 0; col < k / 2; col++) {
        if (col == temp) {
          if (temp & 0x1) {
            hostA[row * k / 2 + col] = 0x98;
          } else {
            hostA[row * k / 2 + col] = 0x89;
          }
        } else {
          hostA[row * k / 2 + col] = 0x88;
        }
      }
    }
  }

  for (int row = 0; row < m; row++) {
    for (int col = 0; col < k / 32; col++) {
      if (row & 0x1) {
        hostQuant[row * k / 32 + col] = 0xbc00;
      } else {
        hostQuant[row * k / 32 + col] = 0x3c00;
      }
    }
  }

  for (int row = 0; row < n; row++) {
    for (int col = 0; col < k; col++) {
      int temp = col & 0x7f;
      hostB[row * k + col] = (float)temp;
    }
  }

  for (int nn = 0; nn < 10; nn++) {
    cudaMalloc(reinterpret_cast<void**>(&deviceA[nn]), sizeA);
    cudaMalloc(reinterpret_cast<void**>(&deviceB[nn]), sizeB);
    cudaMalloc(reinterpret_cast<void**>(&deviceC[nn]), sizeC);
    cudaMemcpyAsync(deviceA[nn], hostA, sizeA, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(deviceB[nn], hostB, sizeB, cudaMemcpyHostToDevice, stream);
  }

  cudaStreamSynchronize(stream);
  // Execute the kernel

  for (int j = 0; j < numbIter; j++) {
    float singleIterTime;
    int bufffIdx = j % 10;
    // Record the start event
    cudaEventRecord(start, stream);
    gemvQ40 <<<grid, threads, 0, stream >>>((uint8_t*)deviceA[bufffIdx], (float*)deviceB[bufffIdx], (float*)deviceC[bufffIdx], m, 1, k);

    cudaEventRecord(stop, stream);
    // Wait for the stop event to complete
    ret = cudaEventSynchronize(stop);
    ret = cudaEventElapsedTime(&singleIterTime, start, stop);
    msecTotal += singleIterTime;
  }

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / numbIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(m) *
                             static_cast<double>(n) *
                             static_cast<double>(k);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

  // Copy result from device to host
  cudaMemcpyAsync(hostC, deviceC[0], sizeC, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  printf("Checking computed result for correctness: ");

  for (int i = 0; i < m * n / 8; i++) {
    for (int j = 0; j < 8; j++) {
      std::cout << hostC[i * 8 + j] << ", ";
    }
    std::cout << "row#" << i << std::endl;
  }

  // Clean up memory
  for (int nn = 0; nn < 10; nn++) {
    cudaFree(deviceA[nn]);
    cudaFree(deviceB[nn]);
    cudaFree(deviceC[nn]);
  }

  cudaFreeHost(hostA);
  cudaFreeHost(hostB);
  cudaFreeHost(hostC);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ret;
}
