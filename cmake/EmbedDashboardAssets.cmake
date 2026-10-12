# EmbedDashboardAssets.cmake - turn a static frontend directory into a C++ source file.
#
# Invoked in script mode:
#   cmake -DINPUT_DIR=<web/dist> -DOUTPUT_FILE=<build/dashboard_assets.cpp>
#         [-DFUNC_PREFIX=dashboard] -P cmake/EmbedDashboardAssets.cmake
#
# FUNC_PREFIX names the generated accessors: "dashboard" (default) produces
# dashboard_asset_count()/dashboard_assets() for the Vue dashboard; "chat"
# produces chat_asset_count()/chat_assets() for the standalone chat UI. Both
# tables share the DashboardAsset type declared in forge/dashboard.h.
#
# Why a generated source file rather than serving from disk:
#   The release tarball ships a bare `forge-cli` binary with no asset directory
#   next to it, and the Docker image copies one file into /usr/local/bin. A
#   runtime lookup would therefore need a --dashboard-root escape hatch and
#   would still break for every installed copy.
#
# Text vs binary assets:
#   html/js/css are emitted as runs of adjacent string literals, which keeps the
#   generated source readable and is what makes the 4000-character chunking
#   necessary (C++ only guarantees 4095 characters per source line and MSVC
#   rejects a single literal above 16380 bytes; ECharts alone pushes the bundle
#   well past that).
#
#   Anything not on the text list is emitted as a byte array instead. A CMake
#   string is NUL-terminated internally, so file(READ) cannot round-trip a
#   binary asset -- a woff2 read this way would arrive truncated. Vite does not
#   emit fonts for this project (the dashboard uses system font stacks), but the
#   byte-array path means adding one later cannot silently corrupt it.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED INPUT_DIR OR NOT DEFINED OUTPUT_FILE)
  message(
    FATAL_ERROR
      "EmbedDashboardAssets.cmake requires -DINPUT_DIR and -DOUTPUT_FILE")
endif()

# Resolve to an absolute path before globbing. In script mode a relative
# INPUT_DIR is resolved against the invoking process's idea of the source dir,
# and a miss produces an empty asset table -- i.e. a dashboard that 404s every
# file with no other symptom. Failing loudly here beats that.
get_filename_component(INPUT_DIR "${INPUT_DIR}" ABSOLUTE)
if(NOT IS_DIRECTORY "${INPUT_DIR}")
  message(
    FATAL_ERROR
      "EmbedDashboardAssets.cmake: INPUT_DIR is not a directory: ${INPUT_DIR}")
endif()

set(_prefix "dashboard")
if(DEFINED FUNC_PREFIX
   AND NOT "${FUNC_PREFIX}" STREQUAL "")
  string(MAKE_C_IDENTIFIER "${FUNC_PREFIX}" _prefix)
endif()

set(_text_exts
    ".html"
    ".js"
    ".mjs"
    ".css"
    ".json"
    ".svg"
    ".map"
    ".txt"
    ".webmanifest")
# Vite emits content-hashed asset names; a source map next to them is pure bulk
# in the binary and useless to anyone debugging a C++ process.
set(_skip_exts ".map")

function(mime_for path out_var)
  get_filename_component(ext "${path}" EXT)
  string(TOLOWER "${ext}" ext)
  if(ext STREQUAL ".html")
    set(m "text/html; charset=utf-8")
  elseif(ext STREQUAL ".js" OR ext STREQUAL ".mjs")
    set(m "text/javascript; charset=utf-8")
  elseif(ext STREQUAL ".css")
    set(m "text/css; charset=utf-8")
  elseif(
    ext STREQUAL ".json"
    OR ext STREQUAL ".webmanifest"
    OR ext STREQUAL ".map")
    set(m "application/json")
  elseif(ext STREQUAL ".svg")
    set(m "image/svg+xml")
  elseif(ext STREQUAL ".png")
    set(m "image/png")
  elseif(ext STREQUAL ".jpg" OR ext STREQUAL ".jpeg")
    set(m "image/jpeg")
  elseif(ext STREQUAL ".gif")
    set(m "image/gif")
  elseif(ext STREQUAL ".ico")
    set(m "image/x-icon")
  elseif(ext STREQUAL ".woff2")
    set(m "font/woff2")
  elseif(ext STREQUAL ".woff")
    set(m "font/woff")
  elseif(ext STREQUAL ".ttf")
    set(m "font/ttf")
  else()
    set(m "application/octet-stream")
  endif()
  set(${out_var}
      "${m}"
      PARENT_SCOPE)
endfunction()

function(in_list needle haystack out_var)
  set(found FALSE)
  foreach(h IN LISTS haystack)
    if(needle STREQUAL h)
      set(found TRUE)
    endif()
  endforeach()
  set(${out_var}
      ${found}
      PARENT_SCOPE)
endfunction()

# Escape a string into a C string body (no chunking). Must be applied to a slice
# that is already self-contained so an escape sequence is never cut in half.
function(escape_body body out_var)
  string(REPLACE "\\" "\\\\" body "${body}")
  string(REPLACE "\"" "\\\"" body "${body}")
  string(REPLACE "\n" "\\n" body "${body}")
  string(REPLACE "\r" "" body "${body}")
  string(REPLACE "\t" "\\t" body "${body}")
  # Escape '?' so the minified JS bundle cannot produce a trigraph sequence.
  # C++17 removed trigraphs, so "??(" is two literal characters rather than "[",
  # but every compiler still warns (-Wtrigraphs) and a warning on a 400 KB
  # single-line literal is noise that hides real ones.
  string(REPLACE "?" "\\?" body "${body}")
  set(${out_var} "${body}" PARENT_SCOPE)
endfunction()

# Escape the text asset and cut it into adjacent string literals. The raw content
# is chunked FIRST (at raw_chunk bytes) and each chunk is escaped in isolation, so
# a backslash in the source can never be split across a literal boundary. The old
# code escaped the whole body then cut it, which left a dangling '\' at the start
# of the next literal whenever a "\\" pair straddled a 4000-char cut -- a hard
# compile error once the bundle grew large enough to land a cut on a backslash.
#
# raw_chunk is the SOURCE size; escaping can at most double length (each byte
# becomes at most "\\"), so 2000 raw bytes stay under the 4095-char single-literal
# limit (and MSVC's 16380) even fully escaped.
function(escape_and_chunk content raw_chunk out_var)
  string(LENGTH "${content}" len)
  if(len EQUAL 0)
    set(${out_var} "\"\"" PARENT_SCOPE)
    return()
  endif()

  set(parts "")
  set(pos 0)
  while(pos LESS len)
    math(EXPR remain "${len} - ${pos}")
    if(remain GREATER raw_chunk)
      set(take ${raw_chunk})
    else()
      set(take ${remain})
    endif()
    string(SUBSTRING "${content}" ${pos} ${take} rawpiece)
    escape_body("${rawpiece}" epiece)
    string(APPEND parts "\"${epiece}\" ")
    math(EXPR pos "${pos} + ${take}")
  endwhile()
  set(${out_var} "${parts}" PARENT_SCOPE)
endfunction()

# Binary: read as HEX (CMake strings survive arbitrary bytes in hex form) and
# emit a byte array. 16 per line keeps the generated file diffable.
function(bytes_array file out_var)
  file(READ "${file}" hexstr HEX)
  string(LENGTH "${hexstr}" hexlen)
  math(EXPR nbytes "${hexlen} / 2")
  set(rows "")
  set(i 0)
  set(_first TRUE)
  while(i LESS nbytes)
    math(EXPR remain "${nbytes} - ${i}")
    if(remain GREATER 16)
      set(take 16)
    else()
      set(take ${remain})
    endif()
    set(row "   ")
    math(EXPR last "${i} + ${take} - 1")
    foreach(k RANGE ${i} ${last})
      math(EXPR byteoff "${k} * 2")
      string(SUBSTRING "${hexstr}" ${byteoff} 2 b)
      # String(TOUPPER) is not available on CMake 3.18; the manual uppercase set
      # below avoids a second pass over a multi-megabyte string.
      string(REPLACE "a" "A" b "${b}")
      string(REPLACE "b" "B" b "${b}")
      string(REPLACE "c" "C" b "${b}")
      string(REPLACE "d" "D" b "${b}")
      string(REPLACE "e" "E" b "${b}")
      string(REPLACE "f" "F" b "${b}")
      string(APPEND row " 0x${b},")
    endforeach()
    string(REGEX REPLACE ",$" "" row "${row}")
    # Rows need a comma BETWEEN them, not at the end of each: stripping every
    # trailing comma (the old behaviour) left "0x00 0x00" with no separator and
    # failed to compile. Keep each row comma-free and join with ",\n".
    if(_first)
      string(APPEND rows "${row}")
      set(_first FALSE)
    else()
      string(APPEND rows ",\n${row}")
    endif()
    math(EXPR i "${i} + ${take}")
  endwhile()
  if(nbytes EQUAL 0)
    set(rows "    0")
  endif()
  set(${out_var}
      "${rows}"
      PARENT_SCOPE)
endfunction()

file(
  GLOB_RECURSE _assets
  RELATIVE "${INPUT_DIR}"
  "${INPUT_DIR}/*")
list(SORT _assets)

set(_out "// Generated by cmake/EmbedDashboardAssets.cmake -- do not edit.\n")
string(APPEND _out "// Source dir: ${INPUT_DIR}\n\n")
string(APPEND _out "#include <cstddef>\n\n")
# DashboardAsset is declared in dashboard.h, which forward-declares httplib::Server
# and so does not pull the HTTP library into this translation unit.
string(APPEND _out "#include \"forge/dashboard.h\"\n\n")
string(APPEND _out "namespace forge {\nnamespace {\n\n")

set(_count 0)
# Per-asset size expression for the table below: text assets are NUL-terminated
# C strings (drop the terminator), binary assets are raw byte arrays (keep all
# bytes). Binary assets define kRaw*, not kAsset*, so the two must not be mixed.
set(_sizes "")
foreach(_rel IN LISTS _assets)
  # GLOB_RECURSE can yield directories on some generators.
  if(IS_DIRECTORY "${INPUT_DIR}/${_rel}")
    continue()
  endif()
  get_filename_component(_ext "${_rel}" EXT)
  string(TOLOWER "${_ext}" _ext)
  in_list("${_ext}" "${_skip_exts}" _skip)
  if(_skip)
    continue()
  endif()
  # Hide dotfiles (.vite/manifest.json and friends).
  get_filename_component(_name "${_rel}" NAME)
  string(SUBSTRING "${_name}" 0 1 _first)
  if(_first STREQUAL ".")
    continue()
  endif()

  math(EXPR _count "${_count} + 1")
  mime_for("${_rel}" _mime)
  in_list("${_ext}" "${_text_exts}" _is_text)

  if(_is_text)
    file(READ "${INPUT_DIR}/${_rel}" _content)
    escape_and_chunk("${_content}" 2000 _body)
    string(APPEND _out "const char kAsset${_count}[] =\n    ${_body};\n")
    string(APPEND _out "const char* const kBody${_count} = kAsset${_count};\n")
  else()
    bytes_array("${INPUT_DIR}/${_rel}" _bytes)
    string(APPEND _out "const unsigned char kRaw${_count}[] = {\n${_bytes}};\n")
    string(
      APPEND
      _out
      "const char* const kBody${_count} = reinterpret_cast<const char*>(kRaw${_count});\n"
    )
  endif()
  if(_is_text)
    list(APPEND _sizes "sizeof(kAsset${_count}) - 1")
  else()
    list(APPEND _sizes "sizeof(kRaw${_count})")
  endif()
  string(APPEND _out "const char kMime${_count}[] = \"${_mime}\";\n")
  string(APPEND _out "const char kPath${_count}[] = \"${_rel}\";\n\n")
  message(STATUS "  embed ${_rel}")
endforeach()

if(_count EQUAL 0)
  message(
    WARNING
      "EmbedDashboardAssets.cmake found no files in ${INPUT_DIR}. The dashboard "
      "will render its 'assets not embedded' page. Did `npm run build` output "
      "somewhere other than dist/?")
endif()

string(APPEND _out "const DashboardAsset kAssets[] = {\n")
if(_count GREATER 0)
  foreach(_i RANGE 1 ${_count})
    math(EXPR _si "${_i} - 1")
    list(GET _sizes ${_si} _sz)
    string(
      APPEND _out
      "    {kPath${_i}, kMime${_i}, kBody${_i}, ${_sz}},\n")
  endforeach()
else()
  # A zero-length array is not valid C++; one placeholder keeps the type valid
  # and dashboard_asset_count() still reports 0.
  string(APPEND _out "    {\"\", \"\", \"\", 0},\n")
endif()
string(APPEND _out "};\n\n")
string(APPEND _out "}  // namespace\n\n")
string(APPEND _out "size_t ${_prefix}_asset_count() { return ${_count}; }\n")
string(APPEND _out
       "const DashboardAsset* ${_prefix}_assets() { return kAssets; }\n\n")
string(APPEND _out "}  // namespace forge\n")
# file(WRITE) with a single quoted argument keeps semicolons, newlines and
# backslashes in the generated body intact, which a list-expanding form mangles.
file(WRITE "${OUTPUT_FILE}" "${_out}")

message(STATUS "Embedded ${_count} dashboard asset(s) -> ${OUTPUT_FILE}")
