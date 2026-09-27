#include "src/core/platform/shutdown_signal.hpp"

#include <cerrno>
#include <csignal>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#include <windows.h>
#else
#include <cstring>
#endif

namespace gufo::platform {
namespace {

#if defined(_WIN32)
using Handler = ShutdownSignalGuard::Handler;
Handler g_active_handler = nullptr;

BOOL WINAPI ConsoleEventRouter(DWORD event) noexcept {
  if (g_active_handler == nullptr)
    return FALSE;
  // Map console events onto their closest POSIX signal numbers so the
  // server sees one uniform "shutdown requested" value.
  const int mapped =
      (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT) ? SIGINT : SIGTERM;
  g_active_handler(mapped);
  return TRUE;
}
#endif

}  // namespace

ShutdownSignalGuard::ShutdownSignalGuard(ShutdownSignalGuard&& other) noexcept
    :
#if !defined(_WIN32)
      previous_interrupt_(other.previous_interrupt_),
      previous_terminate_(other.previous_terminate_),
#endif
      installed_(std::exchange(other.installed_, false)) {
}

ShutdownSignalGuard& ShutdownSignalGuard::operator=(
    ShutdownSignalGuard&& other) noexcept {
  if (this != &other) {
    this->~ShutdownSignalGuard();
    new (this) ShutdownSignalGuard(std::move(other));
  }
  return *this;
}

ShutdownSignalGuard::ShutdownSignalGuard(Handler handler) {
#if defined(_WIN32)
  g_active_handler = handler;
  if (!SetConsoleCtrlHandler(ConsoleEventRouter, TRUE)) {
    const DWORD error = GetLastError();
    g_active_handler = nullptr;
    throw std::system_error(static_cast<int>(error), std::system_category(),
                            "install console ctrl handler");
  }
  installed_ = true;
#else
  struct sigaction action{};
  action.sa_handler = handler;
  ::sigemptyset(&action.sa_mask);
  if (::sigaction(SIGINT, &action, &previous_interrupt_) != 0)
    throw std::system_error(errno, std::generic_category(),
                            "install SIGINT handler");
  if (::sigaction(SIGTERM, &action, &previous_terminate_) != 0) {
    const int error = errno;
    (void)::sigaction(SIGINT, &previous_interrupt_, nullptr);
    throw std::system_error(error, std::generic_category(),
                            "install SIGTERM handler");
  }
  installed_ = true;
#endif
}

ShutdownSignalGuard::~ShutdownSignalGuard() {
#if defined(_WIN32)
  if (installed_) {
    SetConsoleCtrlHandler(ConsoleEventRouter, FALSE);
    g_active_handler = nullptr;
  }
#else
  if (installed_) {
    (void)::sigaction(SIGTERM, &previous_terminate_, nullptr);
    (void)::sigaction(SIGINT, &previous_interrupt_, nullptr);
  }
#endif
}

}  // namespace gufo::platform
