#ifndef GUFO_CORE_PLATFORM_OS_INFO_HPP_
#define GUFO_CORE_PLATFORM_OS_INFO_HPP_

#include <cstddef>
#include <cstdint>

namespace gufo::platform {

/// OS page size in bytes; never zero.
std::size_t PageSizeBytes();

/// Best-effort physically available memory in bytes. Linux reports
/// MemAvailable (sysconf as fallback) clamped by the enclosing cgroup limit;
/// Windows reports GlobalMemoryStatusEx available physical memory.
/// Returns 0 when the OS cannot answer.
std::uint64_t AvailableMemoryBytes();

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_OS_INFO_HPP_
