#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace forge::inspect {

// Report output formats.
enum class Format {
    Text,       // indented key/value + Unicode tree & bar chart (default)
    Table,      // box-drawing aligned tables, width-aware chart
    Json,       // single structured JSON object (no color)
    Yaml,       // nested YAML (no color)
    Markdown,   // markdown tables for each section
    Csv,        // flat tables separated by blank lines (no color)
    Ini,        // strict [section] k = v metadata dump (no color)
};

// "text" -> Format::Text. Returns false for unknown names.
bool format_from_name(const std::string& name, Format& out);
const char* format_name(Format f);

// True for formats where ANSI color must never be emitted.
bool format_is_plain(Format f);

// Renders a box-drawing table with automatic column widths. Numeric cells
// (containing digits or "%" / "B" suffixes) are right-aligned.
std::string box_table(const std::vector<std::string>& headers,
                      const std::vector<std::vector<std::string>>& rows);

// Renders a GitHub-style markdown table.
std::string markdown_table(const std::vector<std::string>& headers,
                           const std::vector<std::vector<std::string>>& rows);

// Renders a CSV table (RFC-4180 quoting; \r\n line endings).
std::string csv_table(const std::vector<std::string>& headers,
                      const std::vector<std::vector<std::string>>& rows);

// 1234 -> "1,234" (thousands separators).
std::string fmt_int(int64_t v);

// Byte count with thousands separators on the numeric part, e.g. "1,234.5 MB".
std::string fmt_bytes(int64_t bytes);

// Percentage with one decimal, e.g "92.3".
std::string fmt_pct(int64_t part, int64_t total);

// Proper terminal width for the default output, or `fallback` when unknown.
int terminal_width(int fallback = 80);

}  // namespace forge::inspect