#ifndef GUFO_TESTS_SUPPORT_TEMPORARY_DIRECTORY_HPP_
#define GUFO_TESTS_SUPPORT_TEMPORARY_DIRECTORY_HPP_

// A unique directory under the temporary path, owner-only where the OS
// supports it, removed when the test ends.

#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

#include "src/core/platform/fs.hpp"
#include "src/core/platform/os_info.hpp"

namespace {

class TemporaryDirectory {
public:
  explicit TemporaryDirectory(std::string name) {
    const auto base = std::filesystem::temp_directory_path();
    for (int attempt = 0; attempt < 64; ++attempt) {
      std::error_code error;
      const auto candidate =
          base / (name + "-" + std::to_string(gufo::platform::ProcessId()) +
                  "-" + std::to_string(attempt));
      if (std::filesystem::create_directory(candidate, error)) {
        if (!gufo::platform::RestrictToOwner(candidate)) {
          throw std::runtime_error("failed to restrict " + name +
                                   " cache directory");
        }
        path_ = candidate;
        return;
      }
    }
    throw std::runtime_error("failed to create " + name + " directory");
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

}  // namespace

#endif  // GUFO_TESTS_SUPPORT_TEMPORARY_DIRECTORY_HPP_
