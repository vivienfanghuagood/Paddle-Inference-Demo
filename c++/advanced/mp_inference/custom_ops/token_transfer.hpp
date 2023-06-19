#pragma once

#include <iostream>
#include <queue>
#include <mutex>
#include <vector>
#include <cstring>
#include <assert.h>


namespace paddle {
namespace inference {
namespace transfer {

struct BatchResult {
  BatchResult(int64_t cur_batch_size, std::vector<int64_t>& cur_tokens):
      batch_size(cur_batch_size), tokens(cur_tokens){}
  int64_t batch_size;
  std::vector<int64_t> tokens;
};


class TokenTransfer {
 public:
  TokenTransfer(const TokenTransfer& o) = delete;
  const TokenTransfer& operator=(const TokenTransfer& o) = delete;
  ~TokenTransfer() {}

  static TokenTransfer &Instance() {
    static TokenTransfer instance;
    return instance;
  }

  // once copy: cpu --> cpu
  // arrary length should be (1 + MAX_BATCH)
  bool GetBatchToken(int64_t* array) {
    if (Empty()) {
      return false;
    } else {
      assert(array != nullptr);
      std::lock_guard<std::mutex> mtx(mtx_);
      array[0] = q_.front().batch_size;
      if (array[0] != 0) {
        memmove(reinterpret_cast<void*>(array + 1), reinterpret_cast<void*>(q_.front().tokens.data()), sizeof(int64_t)* array[0]);
      }
      q_.pop();
      return true;
    }
  }

  void PushBatchToken(int64_t cur_batch_size, int64_t* cur_tokens) {
    std::lock_guard<std::mutex> mtx(mtx_);
    std::vector<int64_t> tmp(cur_tokens, cur_tokens + cur_batch_size);
    q_.emplace(cur_batch_size, tmp);
  }

  bool Empty() {
    std::lock_guard<std::mutex> mtx(mtx_);
    return q_.empty();
  }

 private: 
  TokenTransfer() {}
  
  std::mutex mtx_;
  std::queue<BatchResult> q_;
};


}
}
}