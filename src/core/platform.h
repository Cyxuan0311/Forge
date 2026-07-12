#pragma once
// Cross-platform portability helpers

#ifdef _WIN32
#define NOMINMAX
#include <BaseTsd.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <windows.h>

typedef SSIZE_T ssize_t;
#define PROT_READ 0
#define MAP_PRIVATE 0
#define FORGE_MAP_FAILED ((void*)-1)

// POSIX compat wrappers (avoid macro conflicts with class method names)
static inline int forge_open(const char* path, int flags) {
    return _open(path, flags);
}
static inline int forge_close(int fd) {
    return _close(fd);
}
static inline ssize_t forge_read(int fd, void* buf, size_t count) {
    return static_cast<ssize_t>(_read(fd, buf, static_cast<unsigned int>(count)));
}

struct PortStat {
    long st_size;
};
typedef PortStat forge_stat_t;

static inline int forge_fstat(int fd, forge_stat_t* st) {
    struct _stat64 s;
    int r = _fstat64(fd, &s);
    st->st_size = static_cast<long>(s.st_size);
    return r;
}

static inline void* forge_mmap(void*, size_t length, int, int, int fd, long long) {
    HANDLE hFile = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (hFile == INVALID_HANDLE_VALUE)
        return (void*)-1;
    HANDLE hMap =
        CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, static_cast<DWORD>(length), nullptr);
    if (!hMap)
        return (void*)-1;
    void* ptr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    return ptr ? ptr : (void*)-1;
}

static inline int forge_munmap(void* addr, size_t) {
    return UnmapViewOfFile(addr) ? 0 : -1;
}

#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct stat forge_stat_t;
static inline int forge_fstat(int fd, forge_stat_t* st) {
    return fstat(fd, st);
}
static inline void* forge_mmap(void* a, size_t l, int p, int f, int fd, off_t o) {
    return mmap(a, l, p, f, fd, o);
}
static inline int forge_munmap(void* a, size_t l) {
    return munmap(a, l);
}
#define FORGE_MAP_FAILED MAP_FAILED
static inline int forge_open(const char* path, int flags) {
    return open(path, flags);
}
static inline int forge_close(int fd) {
    return close(fd);
}
static inline ssize_t forge_read(int fd, void* buf, size_t count) {
    return read(fd, buf, count);
}
#endif
