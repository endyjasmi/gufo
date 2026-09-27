#ifndef GUFO_CORE_PLATFORM_PUBLISH_HPP_
#define GUFO_CORE_PLATFORM_PUBLISH_HPP_

#include <cstdint>
#include <filesystem>
#include <string>

namespace gufo::platform {

/// Advisory inter-process exclusion on a lock file (plan database, model
/// single-instance locks). POSIX blocks in flock(LOCK_EX); Windows takes an
/// exclusive byte-range lock. Held until destruction; released by the OS if
/// the process dies. Throws std::system_error when the lock cannot be taken.
class FileLock {
public:
  FileLock() = default;
  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;
  FileLock(FileLock&& other) noexcept;
  FileLock& operator=(FileLock&& other) noexcept;
  ~FileLock();

  static FileLock Acquire(const std::string& path);

private:
  explicit FileLock(void* handle) noexcept : handle_(handle) {}

  void Release() noexcept;

  void* handle_ = nullptr;  // POSIX: int fd stored as void*; Windows: HANDLE
};

/// A temporary file created next to its final destination, published by
/// Commit() with flush-then-rename semantics: fsync, rename over the
/// destination, then fsync the containing directory (POSIX). On Windows the
/// temp file is flushed and moved with MOVEFILE_REPLACE_EXISTING; NTFS
/// journaling makes the directory flush redundant.
class AtomicPublishedFile {
public:
  AtomicPublishedFile() = default;
  AtomicPublishedFile(const AtomicPublishedFile&) = delete;
  AtomicPublishedFile& operator=(const AtomicPublishedFile&) = delete;
  AtomicPublishedFile(AtomicPublishedFile&& other) noexcept;
  AtomicPublishedFile& operator=(AtomicPublishedFile&& other) noexcept;
  ~AtomicPublishedFile();

  /// Creates `<final_path>.tmp.XXXXXX` in the destination directory.
  /// Returns an invalid file and sets `error` on failure.
  static AtomicPublishedFile Create(const std::filesystem::path& final_path,
                                    std::string* error);

  bool valid() const noexcept { return handle_ != nullptr; }

  bool Write(const void* data, std::size_t bytes, std::string* error);

  /// Flush, replace the destination, and flush the directory. Returns false
  /// and sets `error` on failure; the temporary file is removed either way.
  bool Commit(const std::filesystem::path& final_path, std::string* error);

private:
  AtomicPublishedFile(void* handle, const std::filesystem::path& temporary)
      : handle_(handle), temporary_(temporary) {}

  void Discard() noexcept;

  void* handle_ = nullptr;  // POSIX: int fd stored as void*; Windows: HANDLE
  std::filesystem::path temporary_;
};

}  // namespace gufo::platform

#endif  // GUFO_CORE_PLATFORM_PUBLISH_HPP_
