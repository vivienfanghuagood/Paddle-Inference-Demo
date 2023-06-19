#include <thread>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <time.h>


#include "token_transfer.hpp"

using namespace paddle::inference::transfer;

#define TEST_TIME 10000

void PrintVec(std::vector<int64_t> & vec) {
  // return;
  std::cout << "000 vec_size: " << vec.size();
  for(int i {0}; i < vec.size(); i++) {
    std::cout << " " << vec[i];
  }
  std::cout << std::endl;
}

void PrintVec(int64_t *arr) {
  std::cout << "READ vec_size: " << arr[0];
  for(int i {1}; i < arr[0] + 1; i++) {
    std::cout << " " << arr[i];
  }
  std::cout << std::endl;
}

void PrintVec(int64_t bs, int64_t *arr) {
  std::cout << "WRITE vec_size: " << bs;
  for(int i {0}; i < bs; i++) {
    std::cout << " " << arr[i];
  }
  std::cout << std::endl;
}

void Write() {
  using namespace std;
  // srand((unsigned)time(nullptr));
  for (int i{0}; i < TEST_TIME; i++) {
    int64_t batch_size = rand() % 20;
    int64_t tokens[batch_size];
    for (int j{0}; j < batch_size; j++) {
      tokens[j] = (rand() % 100);
    }
    PrintVec(batch_size, tokens);
    TokenTransfer::Instance().PushBatchToken(batch_size, tokens);
  }

}

void Read() {
  using namespace std;
  for (int i{0}; i < TEST_TIME; i++) {
    int64_t result[21];
    while (!TokenTransfer::Instance().GetBatchToken(result));
    PrintVec(result);
  }
  cout << "finish" << endl;
}


int main() {
  auto start = std::chrono::system_clock::now();
  std::thread read_thread{Read};
  std::thread wirte_thread{Write};
  read_thread.join();
  wirte_thread.join();
  auto sec = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);
  std::cout << "duration: " << sec.count() << std::endl;
  return 0;
}