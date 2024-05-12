#include "../kernel/gemv.cuh"
#include "../include/gemv.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
  int m, n, k;
  int numbIter;
  testId test;
  if (argc == 0) {
    std::cout << "usage: gemv.exe [testID]" << std::endl;
    return -1;
  }

  test = (testId)atoi(argv[1]);
  switch (test) {
  case testIdGemvQ40:
    m = k = 4096;
    numbIter = 1000;
    if (argc > 2) {
      k = atoi(argv[2]);
    }

    if (argc > 3) {
      m = atoi(argv[3]);
    }

    if (argc > 4) {
      numbIter = atoi(argv[4]);
    }
    testGemvQ40(m, 1, k, numbIter);
    break;
  default:
    break;
  }

  return 0;
}