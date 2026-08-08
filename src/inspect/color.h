#pragma once

#include <string>

namespace forge::inspect::col {

// ANSI text styles used across forge-inspect's renderers.
enum class Style {
    Reset,
    Bold,
    Dim,
    Underline,
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
    BoldRed,
    BoldGreen,
    BoldYellow,
    BoldBlue,
    BoldMagenta,
    BoldCyan,
    DimRed,
    DimGreen,
    DimYellow,
    DimBlue,
    DimMagenta,
    DimCyan,
};

// Color policy. Auto (default) enables color only when stdout is a TTY and
// NO_COLOR is not set.
enum class Mode { Auto, Always, Never };

void set_mode(Mode m);
Mode mode();

// Re-evaluates the effective flag (TTY + NO_COLOR) for Auto mode.
void refresh();

bool enabled();

// Wraps `s` in ANSI escape codes when color is enabled; returns `s` untouched
// otherwise (so file output stays clean).
std::string paint(std::string s, Style st);

}  // namespace forge::inspect::col
