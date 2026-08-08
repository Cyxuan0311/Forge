#include "color.h"

#include <cstdlib>

#ifdef _WIN32
#include <io.h>
#define ISATTY(fd) _isatty(_fileno(fd))
#else
#include <unistd.h>
#define ISATTY(fd) isatty(fd)
#endif

namespace forge::inspect::col {

namespace {

Mode g_mode = Mode::Auto;
bool g_enabled = false;

const char* code(Style st) {
    switch (st) {
        case Style::Reset:        return "\033[0m";
        case Style::Bold:         return "\033[1m";
        case Style::Dim:          return "\033[2m";
        case Style::Underline:    return "\033[4m";
        case Style::Black:        return "\033[30m";
        case Style::Red:          return "\033[31m";
        case Style::Green:        return "\033[32m";
        case Style::Yellow:       return "\033[33m";
        case Style::Blue:         return "\033[34m";
        case Style::Magenta:      return "\033[35m";
        case Style::Cyan:         return "\033[36m";
        case Style::White:        return "\033[37m";
        case Style::BoldRed:      return "\033[1;31m";
        case Style::BoldGreen:    return "\033[1;32m";
        case Style::BoldYellow:   return "\033[1;33m";
        case Style::BoldBlue:     return "\033[1;34m";
        case Style::BoldMagenta:  return "\033[1;35m";
        case Style::BoldCyan:     return "\033[1;36m";
        case Style::DimRed:       return "\033[2;31m";
        case Style::DimGreen:     return "\033[2;32m";
        case Style::DimYellow:    return "\033[2;33m";
        case Style::DimBlue:      return "\033[2;34m";
        case Style::DimMagenta:   return "\033[2;35m";
        case Style::DimCyan:      return "\033[2;36m";
    }
    return "";
}

}  // namespace

void refresh() {
    if (g_mode == Mode::Always) {
        g_enabled = true;
        return;
    }
    if (g_mode == Mode::Never) {
        g_enabled = false;
        return;
    }
    const char* nc = std::getenv("NO_COLOR");
    g_enabled = !(nc && *nc) && ISATTY(1) != 0;
}

void set_mode(Mode m) {
    g_mode = m;
    refresh();
}

Mode mode() {
    return g_mode;
}

bool enabled() {
    return g_enabled;
}

std::string paint(std::string s, Style st) {
    if (!g_enabled || s.empty())
        return s;
    std::string out = code(st);
    out += s;
    out += code(Style::Reset);
    return out;
}

}  // namespace forge::inspect::col
