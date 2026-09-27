#ifndef GUFO_CORE_PLATFORM_NET_HPP_
#define GUFO_CORE_PLATFORM_NET_HPP_

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace gufo::net {

#if defined(_WIN32)
using Socket = SOCKET;
#else
using Socket = int;
#endif

/// The one-time WinSock startup the HTTP server performs before opening its
/// listener; a no-op on POSIX. Throws std::system_error on failure.
void EnsureSocketRuntimeStarted();

/// Close a socket: closesocket on Windows, close on POSIX.
void CloseSocket(Socket socket) noexcept;

/// poll()/WSAPoll() with the platform's native timeout unit.
int PollSocket(void* descriptors, std::uint32_t count, int timeout_ms);

/// Send that never raises SIGPIPE (MSG_NOSIGNAL on POSIX; default on Winsock).
std::intptr_t SendNoSignal(Socket socket, const void* data, std::size_t bytes);

/// Mark a socket non-blocking; returns false on failure.
bool SetNonBlocking(Socket socket) noexcept;

/// Disable Nagle on a connected stream socket; returns false on failure.
bool SetNoDelay(Socket socket) noexcept;

/// Set receive/send timeouts in milliseconds; returns false on failure.
bool SetSocketTimeouts(Socket socket, int timeout_ms) noexcept;

}  // namespace gufo::net

#endif  // GUFO_CORE_PLATFORM_NET_HPP_
