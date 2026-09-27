#include "src/core/platform/mapped_file.hpp"

#include <algorithm>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace gufo::platform {
namespace {

#if defined(_WIN32)
std::string FormatWindowsError(const char* what, DWORD code) {
  return std::string(what) + " failed with Windows error " +
         std::to_string(code);
}
#endif

}  // namespace

ReadOnlyMappedFile::ReadOnlyMappedFile(ReadOnlyMappedFile&& other) noexcept
    : ReadOnlyMappedFile(other.data_, other.size_, other.handle_,
                         other.mapping_) {
  other.data_ = nullptr;
  other.size_ = 0;
  other.handle_ = nullptr;
  other.mapping_ = nullptr;
}

ReadOnlyMappedFile& ReadOnlyMappedFile::operator=(
    ReadOnlyMappedFile&& other) noexcept {
  if (this != &other) {
    Close();
    data_ = other.data_;
    size_ = other.size_;
    handle_ = other.handle_;
    mapping_ = other.mapping_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.handle_ = nullptr;
    other.mapping_ = nullptr;
  }
  return *this;
}

ReadOnlyMappedFile::~ReadOnlyMappedFile() {
  Close();
}

int ReadOnlyMappedFile::PlatformDescriptor() const noexcept {
#if !defined(_WIN32)
  return handle_ != nullptr
             ? static_cast<int>(reinterpret_cast<std::uintptr_t>(handle_))
             : -1;
#else
  return -1;
#endif
}

void ReadOnlyMappedFile::Close() noexcept {
#if defined(_WIN32)
  if (data_ != nullptr)
    UnmapViewOfFile(data_);
  if (mapping_ != nullptr)
    CloseHandle(static_cast<HANDLE>(mapping_));
  if (handle_ != nullptr)
    CloseHandle(static_cast<HANDLE>(handle_));
#else
  if (data_ != nullptr)
    munmap(const_cast<std::uint8_t*>(data_), size_);
  if (handle_ != nullptr)
    ::close(static_cast<int>(reinterpret_cast<std::uintptr_t>(handle_)));
#endif
  data_ = nullptr;
  size_ = 0;
  handle_ = nullptr;
  mapping_ = nullptr;
}

ReadOnlyMappedFile ReadOnlyMappedFile::Open(const std::string& path,
                                            std::string* error) {
  const auto fail = [&](std::string message) {
    if (error != nullptr)
      *error = std::move(message);
    return ReadOnlyMappedFile();
  };
#if defined(_WIN32)
  const HANDLE file =
      CreateFileA(path.c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING,
                  FILE_ATTRIBUTE_READONLY | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    return fail(FormatWindowsError("CreateFileA", GetLastError()));
  LARGE_INTEGER file_size{};
  if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart < 0) {
    const DWORD code = GetLastError();
    CloseHandle(file);
    return fail(FormatWindowsError("GetFileSizeEx", code));
  }
  const HANDLE mapping =
      CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    const DWORD code = GetLastError();
    CloseHandle(file);
    return fail(FormatWindowsError("CreateFileMappingA", code));
  }
  void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    const DWORD code = GetLastError();
    CloseHandle(mapping);
    CloseHandle(file);
    return fail(FormatWindowsError("MapViewOfFile", code));
  }
  return ReadOnlyMappedFile(static_cast<const std::uint8_t*>(view),
                            static_cast<std::size_t>(file_size.QuadPart), file,
                            mapping);
#else
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return fail("cannot open " + path + ": " + std::to_string(errno));
  struct stat st{};
  if (::fstat(fd, &st) != 0 || st.st_size < 0) {
    const int code = errno;
    ::close(fd);
    return fail("cannot stat " + path + ": " + std::to_string(code));
  }
  if (st.st_size == 0) {
    ::close(fd);
    return fail("file is empty: " + path);
  }
  void* view = ::mmap(nullptr, static_cast<std::size_t>(st.st_size), PROT_READ,
                      MAP_PRIVATE, fd, 0);
  if (view == MAP_FAILED) {
    const int code = errno;
    ::close(fd);
    return fail("cannot mmap " + path + ": " + std::to_string(code));
  }
  ::madvise(view, static_cast<std::size_t>(st.st_size), MADV_SEQUENTIAL);
  return ReadOnlyMappedFile(
      static_cast<const std::uint8_t*>(view),
      static_cast<std::size_t>(st.st_size),
      reinterpret_cast<void*>(static_cast<std::uintptr_t>(fd)), nullptr);
#endif
}

std::size_t SystemPageSize() noexcept {
#if defined(_WIN32)
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  return info.dwPageSize != 0 ? static_cast<std::size_t>(info.dwPageSize)
                              : static_cast<std::size_t>(4096);
#else
  const long page_size = ::sysconf(_SC_PAGESIZE);
  return page_size > 0 ? static_cast<std::size_t>(page_size)
                       : static_cast<std::size_t>(4096);
#endif
}

bool RawMappedFile::Open(const char* path) noexcept {
#if defined(_WIN32)
  const HANDLE file =
      CreateFileA(path, GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING,
                  FILE_ATTRIBUTE_READONLY | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    return false;
  LARGE_INTEGER file_size{};
  if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart <= 0) {
    CloseHandle(file);
    return false;
  }
  const HANDLE section =
      CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (section == nullptr) {
    CloseHandle(file);
    return false;
  }
  void* view = MapViewOfFile(section, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    CloseHandle(section);
    CloseHandle(file);
    return false;
  }
  data = view;
  size = static_cast<std::size_t>(file_size.QuadPart);
  this->file = file;
  this->section = section;
  return true;
#else
  const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return false;
  struct stat status{};
  if (::fstat(fd, &status) != 0 || status.st_size <= 0) {
    ::close(fd);
    return false;
  }
  void* mapping = ::mmap(nullptr, static_cast<std::size_t>(status.st_size),
                         PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (mapping == MAP_FAILED)
    return false;
  ::madvise(mapping, static_cast<std::size_t>(status.st_size), MADV_SEQUENTIAL);
  data = mapping;
  size = static_cast<std::size_t>(status.st_size);
  return true;
#endif
}

bool RawMappedFile::OpenRange(const char* path, std::uint64_t offset,
                              std::size_t length) noexcept {
#if defined(_WIN32)
  const HANDLE file =
      CreateFileA(path, GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING,
                  FILE_ATTRIBUTE_READONLY | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    return false;
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  const std::uint64_t granularity =
      info.dwAllocationGranularity != 0 ? info.dwAllocationGranularity : 65536;
  const std::uint64_t base = offset - (offset % granularity);
  const std::uint64_t span = offset - base + length;
  const HANDLE section =
      CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (section == nullptr) {
    CloseHandle(file);
    return false;
  }
  void* view =
      MapViewOfFile(section, FILE_MAP_READ, static_cast<DWORD>(base >> 32),
                    static_cast<DWORD>(base), static_cast<std::size_t>(span));
  if (view == nullptr) {
    CloseHandle(section);
    CloseHandle(file);
    return false;
  }
  data = static_cast<std::uint8_t*>(view) + (offset - base);
  size = length;
  this->file = file;
  this->section = section;
  view_base = view;
  view_size = static_cast<std::size_t>(span);
  return true;
#else
  const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return false;
  void* mapping = ::mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd,
                         static_cast<off_t>(offset));
  ::close(fd);
  if (mapping == MAP_FAILED)
    return false;
  ::madvise(mapping, length, MADV_SEQUENTIAL);
  data = mapping;
  size = length;
  return true;
#endif
}

void RawMappedFile::Close() noexcept {
#if defined(_WIN32)
  if (view_base != nullptr)
    UnmapViewOfFile(view_base);
  else if (data != nullptr)
    UnmapViewOfFile(data);
  if (section != nullptr)
    CloseHandle(static_cast<HANDLE>(section));
  if (file != nullptr)
    CloseHandle(static_cast<HANDLE>(file));
  file = nullptr;
  section = nullptr;
  view_base = nullptr;
  view_size = 0;
#else
  if (data != nullptr)
    munmap(data, size);
#endif
  data = nullptr;
  size = 0;
}

PlatformFile OpenReadFile(const std::string& path) noexcept {
#if defined(_WIN32)
  return CreateFileA(path.c_str(), GENERIC_READ,
                     FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                     nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
#else
  return ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
}

bool FileValid(PlatformFile file) noexcept {
#if defined(_WIN32)
  return file != nullptr && file != INVALID_HANDLE_VALUE;
#else
  return file >= 0;
#endif
}

void CloseFile(PlatformFile file) noexcept {
  if (!FileValid(file))
    return;
#if defined(_WIN32)
  CloseHandle(static_cast<HANDLE>(file));
#else
  ::close(file);
#endif
}

std::size_t ReadFileAt(PlatformFile file, void* output, std::size_t bytes,
                       std::uint64_t offset) noexcept {
#if defined(_WIN32)
  OVERLAPPED overlapped{};
  overlapped.Offset = static_cast<DWORD>(offset);
  overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
  DWORD read_bytes = 0;
  if (!ReadFile(static_cast<HANDLE>(file), output, static_cast<DWORD>(bytes),
                &read_bytes, &overlapped)) {
    return 0;
  }
  return read_bytes;
#else
  return static_cast<std::size_t>(
      ::pread(file, output, bytes, static_cast<off_t>(offset)));
#endif
}

std::optional<std::uint64_t> FileSizeBytes(PlatformFile file) noexcept {
#if defined(_WIN32)
  LARGE_INTEGER size{};
  if (!GetFileSizeEx(static_cast<HANDLE>(file), &size) || size.QuadPart < 0)
    return std::nullopt;
  return static_cast<std::uint64_t>(size.QuadPart);
#else
  struct stat status{};
  if (::fstat(file, &status) != 0 || status.st_size < 0)
    return std::nullopt;
  return static_cast<std::uint64_t>(status.st_size);
#endif
}

}  // namespace gufo::platform
