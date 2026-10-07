#include "src/cli/serve/trace.hpp"

#if !defined(_WIN32)
#include <sys/resource.h>
#include <unistd.h>
#else
#include <process.h>
// Upstream's POSIX environment helper, for the NO_COLOR setup.
static void setenv(const char* name, const char* value, int) {
  (void)_putenv_s(name, value);
}
#endif

#include <cassert>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/core/json.hpp"

namespace {

using gufo::server::Trace;
namespace fs = std::filesystem;

void WriteRecord(std::string_view event, std::string_view text) {
  auto record = Trace::Record(event, "r1");
  record["text"] = std::string(text);
  Trace::Write(record);
}

std::vector<std::string> Lines(const fs::path& path) {
  std::vector<std::string> lines;
  std::ifstream input(path);
  for (std::string line; std::getline(input, line);) {
    lines.push_back(line);
  }
  return lines;
}

#if !defined(_WIN32)
// A record cut short by a full disk or file-size limit is removed, so the next
// record still starts its own line and the file stays valid JSON Lines.
void TestPartialWriteIsRolledBack() {
  const auto path =
      fs::temp_directory_path() /
      ("gufo-trace-partial-" + std::to_string(::getpid()) + ".jsonl");
  fs::remove(path);
  assert(!Trace::Open(path.string()).has_value());
  WriteRecord("before", "kept");
  const auto kept_bytes = fs::file_size(path);

  // Past the limit, write() appends what fits and then fails with EFBIG; the
  // signal would otherwise end the process.
  (void)std::signal(SIGXFSZ, SIG_IGN);
  rlimit original{};
  assert(::getrlimit(RLIMIT_FSIZE, &original) == 0);
  rlimit limited = original;
  limited.rlim_cur = kept_bytes + 32;
  assert(::setrlimit(RLIMIT_FSIZE, &limited) == 0);
  WriteRecord("torn", std::string(1024, 'x'));
  assert(::setrlimit(RLIMIT_FSIZE, &original) == 0);
  assert(fs::file_size(path) == kept_bytes);

  assert(Trace::Enabled());
  WriteRecord("after", "recovered");
  Trace::Close();

  const auto lines = Lines(path);
  fs::remove(path);
  assert(lines.size() == 2);
  assert(gufo::json::parse(lines[0]).member_str("event") == "before");
  assert(gufo::json::parse(lines[1]).member_str("event") == "after");
}
#endif

}  // namespace

int main() {
  ::setenv("NO_COLOR", "1", 1);
#if !defined(_WIN32)
  TestPartialWriteIsRolledBack();
  std::cout << "Trace sink checks passed.\n";
#else
  // RLIMIT_FSIZE has no Windows equivalent, so the torn-record rollback
  // cannot be simulated here; exercise the open/append/close path instead.
  const auto path =
      fs::temp_directory_path() /
      ("gufo-trace-basic-" + std::to_string(_getpid()) + ".jsonl");
  fs::remove(path);
  assert(!Trace::Open(path.string()).has_value());
  WriteRecord("before", "kept");
  assert(Trace::Enabled());
  Trace::Close();
  assert(!Trace::Enabled());

  const auto lines = Lines(path);
  fs::remove(path);
  assert(lines.size() == 1);
  assert(gufo::json::parse(lines[0]).member_str("event") == "before");
  std::cout << "Trace sink open/append/close checks passed; the torn-record "
               "rollback path needs POSIX rlimits.\n";
#endif
}
