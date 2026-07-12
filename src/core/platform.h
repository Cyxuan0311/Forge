#pragma once
// Cross-platform portability helpers

#ifdef _WIN32
// MSVC POSIX compat
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#define open _open
#define close _close
#define read _read
#define O_RDONLY _O_RDONLY
#define PROT_READ 0
#define MAP_PRIVATE 0

#include <io.h>
#include <sys/stat.h>
#include <windows.h>

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

#define FORGE_MAP_FAILED ((void*)-1)

#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct stat forge_stat_t;
#define forge_fstat fstat
#define forge_mmap mmap
#define forge_munmap munmap
#define FORGE_MAP_FAILED MAP_FAILED
#endif
