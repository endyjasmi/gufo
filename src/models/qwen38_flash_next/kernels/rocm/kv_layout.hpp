// KV cache element formats. The F16 cache stores one __half per element;
// the Q8_0 cache stores GGML-style 32-element blocks with the F32→int8
// codes packed first and their F16 scales trailing, so vector loads of the
// codes stay aligned and the dequant multiply is one scale per block:
// [kv_row int8][kv_row/32 F16 scales] = kv_row + kv_row/16 bytes per row.
#pragma once

#include <cstddef>
#include <cstdint>

namespace gufo::models::qwen38_flash_next {

enum class KvCacheDtype : std::uint32_t {
  kF16 = 0,
  kQ8_0 = 1,
};

constexpr std::uint32_t kKvBlock = 32;  // elements per Q8_0 scale block

inline constexpr std::size_t KvRowBytes(std::size_t kv_row_elems,
                                        KvCacheDtype dtype) {
  return dtype == KvCacheDtype::kQ8_0
             ? kv_row_elems + (kv_row_elems / kKvBlock) * 2
             : kv_row_elems * 2;
}

}  // namespace gufo::models::qwen38_flash_next
