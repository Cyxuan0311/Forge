#include "format.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace forge::inspect {

bool format_from_name(const std::string& name, Format& out) {
    if (name == "text") out = Format::Text;
    else if (name == "table") out = Format::Table;
    else if (name == "json") out = Format::Json;
    else if (name == "yaml") out = Format::Yaml;
    else if (name == "markdown" || name == "md") out = Format::Markdown;
    else if (name == "csv") out = Format::Csv;
    else if (name == "ini") out = Format::Ini;
    else
        return false;
    return true;
}

const char* format_name(Format f) {
    switch (f) {
        case Format::Text: return "text";
        case Format::Table: return "table";
        case Format::Json: return "json";
        case Format::Yaml: return "yaml";
        case Format::Markdown: return "markdown";
        case Format::Csv: return "csv";
        case Format::Ini: return "ini";
    }
    return "text";
}

bool is_plain(Format f) {
    return f == Format::Json || f == Format::Yaml || f == Format::Csv ||
           f == Format::Ini;
}

std::string fmt_int(int64_t v) {
    const bool neg = v < 0;
    const unsigned long long a = neg ? static_cast<unsigned long long>(-(v + 1)) + 1
                                     : static_cast<unsigned long long>(v);
    std::string s = std::to_string(a);
    std::string out;
    int cnt = 0;
    for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i) {
        out.push_back(s[i]);
        if (++cnt % 3 == 0 && i > 0)
            out.push_back(',');
    }
    std::reverse(out.begin(), out.end());
    return neg ? "-" + out : out;
}

std::string fmt_bytes(int64_t bytes) {
    static const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double v = static_cast<double>(bytes < 0 ? -bytes : bytes);
    int u = 0;
    while (v >= 1024.0 && u < 4) {
        v /= 1024.0;
        ++u;
    }
    char nbuf[32];
    std::snprintf(nbuf, sizeof(nbuf), "%.1f", v);
    std::string num = nbuf;
    // Insert thousands separators into the integer part.
    size_t dot = num.find('.');
    std::string ip = dot == std::string::npos ? num : num.substr(0, dot);
    std::string grouped;
    int cnt = 0;
    for (int i = static_cast<int>(ip.size()) - 1; i >= 0; --i) {
        grouped.insert(grouped.begin(), ip[i]);
        if (++cnt % 3 == 0 && i > 0)
            grouped.insert(grouped.begin(), ',');
    }
    num = dot == std::string::npos ? grouped : grouped + num.substr(dot);
    return num + " " + units[u];
}

std::string fmt_pct(int64_t part, int64_t total) {
    if (total <= 0)
        return "0.0";
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.1f",
                  100.0 * static_cast<double>(part) / static_cast<double>(total));
    return buf;
}

namespace {

// Display width of a UTF-8 string (CJK glyphs approximated as 2 columns).
int cell_width(const std::string& s) {
    int w = 0;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        int len = 1;
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        const int seq = (len == 1) ? 2 : len;  // CJK sets {F0,E0} wide
        const bool wide = (len == 3) || (len == 4) || (c == 0xC2 && i + 1 < s.size() && s[i+1] >= 0x80);
        (void)seq;
        if (i + len > s.size())
            len = static_cast<int>(s.size()) - static_cast<int>(i);
        if (len <= 0)
            break;
        ++w;
        if (wide)
            ++w;
        i += static_cast<size_t>(len);
    }
    return w;
}

bool is_numeric(const std::string& s) {
    const char* allow = " ,.%+-BKMGTPIE";
    for (char c : s) {
        if (c >= '0' && c <= '9')
            continue;
        bool ok = false;
        for (const char* p = allow; *p; ++p)
            if (c == *p) { ok = true; break; }
        if (!ok)
            return false;
    }
    return !s.empty();
}

std::string pad_cell(const std::string& s, int width, bool right) {
    const int pad = width - cell_width(s);
    if (pad <= 0)
        return s;
    return right ? std::string(static_cast<size_t>(pad), ' ') + s
                 : s + std::string(static_cast<size_t>(pad), ' ');
}

std::string border(const std::string& left, const std::string& connector,
                   const std::string& right, const std::string& fill,
                   const std::vector<int>& widths) {
    std::string out = left;
    for (size_t c = 0; c < widths.size(); ++c) {
        out += std::string(static_cast<size_t>(widths[c]) + 2, fill[0]);
        out += (c + 1 < widths.size()) ? connector : right;
    }
    return out + "\n";
}

std::string data_row(const std::string& sep, const char* edge,
                     const std::vector<int>& widths,
                     const std::vector<std::string>& cells) {
    std::string out = edge;
    for (size_t c = 0; c < widths.size(); ++c) {
        const std::string cell = c < cells.size() ? cells[c] : "";
        out += " " + pad_cell(cell, widths[c], is_numeric(cell)) + " ";
        out += (c + 1 < widths.size()) ? sep : edge;
    }
    return out + "\n";
}

}  // namespace

std::string box_table(const std::vector<std::string>& headers,
                      const std::vector<std::vector<std::string>>& rows) {
    if (headers.empty())
        return "";
    const size_t ncols = headers.size();
    std::vector<int> widths(ncols, 0);
    for (size_t c = 0; c < ncols; ++c)
        widths[c] = std::max(widths[c], cell_width(headers[c]));
    for (const auto& row : rows)
        for (size_t c = 0; c < ncols && c < row.size(); ++c)
            widths[c] = std::max(widths[c], cell_width(row[c]));

    std::string out = border("┌", "┬", "┐", "─", widths);
    out += data_row("│", "│", widths, headers);
    out += border("├", "┼", "┤", "─", widths);
    for (const auto& row : rows)
        out += data_row("│", "│", widths, row);
    out += border("└", "┴", "┘", "─", widths);
    return out;
}

std::string markdown_table(const std::vector<std::string>& headers,
                           const std::vector<std::vector<std::string>>& rows) {
    if (headers.empty())
        return "";
    auto esc = [](std::string s) {
        for (size_t i = 0; i < s.size(); ++i)
            if (s[i] == '|')
                s[i] = '\\';
        return s;
    };
    std::string out;
    const auto header_row = [&](const std::vector<std::string>& cells) {
        std::string r = "|";
        for (const auto& c : cells)
            r += " " + esc(c) + " |";
        return r;
    };
    out += header_row(headers) + "\n|";
    for (size_t c = 0; c < headers.size(); ++c)
        out += " --- |";
    out += "\n";
    for (const auto& row : rows)
        out += header_row(row) + "\n";
    return out;
}

namespace {
std::string csv_quote(const std::string& s) {
    bool need = s.find_first_of(",\"\r\n") != std::string::npos;
    if (!need)
        return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"')
            out += "\"\"";
        else
            out += c;
    }
    out += '"';
    return out;
}
}  // namespace

std::string csv_table(const std::vector<std::string>& headers,
                      const std::vector<std::vector<std::string>>& rows) {
    const auto row_str = [](const std::vector<std::string>& cells) {
        std::string out;
        for (size_t c = 0; c < cells.size(); ++c) {
            if (c)
                out += ",";
            out += csv_quote(cells[c]);
        }
        return out;
    };
    std::string out = row_str(headers) + "\r\n";
    for (const auto& row : rows)
        out += row_str(row) + "\r\n";
    return out;
}

int terminal_width(int fallback) {
    if (const char* c = std::getenv("COLUMNS")) {
        long v = std::strtol(c, nullptr, 10);
        if (v > 40)
            return static_cast<int>(v);
    }
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
        const int n = csbi.srWindow.Right - csbi.srWindow.Left + 1;
        if (n > 40)
            return n;
    }
#else
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 40)
        return ws.ws_col;
#endif
    return fallback;
}

}  // namespace forge::inspect