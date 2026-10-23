#ifndef GUFO_CORE_PLATFORM_PREFETCH_HPP_
#define GUFO_CORE_PLATFORM_PREFETCH_HPP_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace gufo::platform {

/// Populate an existing readable mapping before HIP registration or upload.
/// Parallel faults expose disk queue depth without making another weight copy.
/// The mapping must remain alive until this call returns, including on failure.
///
/// Linux faults pages with MADV_POPULATE_READ from several threads; Windows
/// asks the memory manager to prefetch the range into the standby list and
/// then touches pages in parallel so failures surface before registration.
inline void PrefaultMappedRange(const void* data, std::size_t bytes) {
  if (bytes == 0)
    return;
#if defined(_WIN32)
  const std::size_t page_size = [] {
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return info.dwPageSize != 0 ? static_cast<std::size_t>(info.dwPageSize)
                                : static_cast<std::size_t>(4096);
  }();
#else
  const long page_size = ::sysconf(_SC_PAGESIZE);
#endif
  if (!data || page_size == 0)
    throw std::invalid_argument("invalid mapped weight range");
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - address)
    throw std::length_error("mapped weight address overflows");
  const auto skip = address % static_cast<std::size_t>(page_size);
  if (bytes > std::numeric_limits<std::size_t>::max() - skip)
    throw std::length_error("mapped weight range overflows");
  const auto begin = address - skip;
  bytes += skip;
  constexpr std::size_t kChunkBytes = 16ULL << 20;
  constexpr std::size_t kReaders = 16;
  const auto chunks = bytes / kChunkBytes + (bytes % kChunkBytes != 0);
  std::atomic<std::size_t> next{0};
  std::atomic<int> failure{0};
#if defined(_WIN32)
  // Windows 8+; returns immediately, prefetching into the standby list.
  WIN32_MEMORY_RANGE_ENTRY entry{reinterpret_cast<void*>(begin), bytes};
  PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
  volatile const std::uint8_t* touch =
      reinterpret_cast<volatile const std::uint8_t*>(begin);
#endif
  auto populate = [&] {
    while (failure.load(std::memory_order_relaxed) == 0) {
      const auto index = next.fetch_add(1, std::memory_order_relaxed);
      if (index >= chunks)
        break;
      const auto offset = index * kChunkBytes;
      const auto span = std::min(kChunkBytes, bytes - offset);
#if defined(_WIN32)
      // Confirm the prefetched pages are resident and readable.
      for (std::size_t i = 0; i < span; i += page_size)
        static_cast<void>(touch[offset + i]);
#else
      if (::madvise(reinterpret_cast<void*>(begin + offset), span,
                    MADV_POPULATE_READ) != 0)
        failure.store(errno, std::memory_order_relaxed);
#endif
    }
  };
  {
    std::vector<std::jthread> readers;
    for (std::size_t i = 1; i < std::min(kReaders, chunks); ++i)
      readers.emplace_back(populate);
    populate();
    // Join before a caller can register, upload, or unmap any of these pages.
  }
  if (const int code = failure.load(); code != 0)
    throw std::system_error(code, std::generic_category(),
                            "cannot read mapped weights");
}

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_PREFETCH_HPP_
