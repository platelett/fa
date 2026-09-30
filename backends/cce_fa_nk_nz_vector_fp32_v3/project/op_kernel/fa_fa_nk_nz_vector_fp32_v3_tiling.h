#pragma once
#include <cstdint>
struct FaFaNkNzVectorFp32V3TilingData {
  uint32_t batch, heads_q, heads_kv, nq, nk, q_l1, groups, capture, phase;
};
