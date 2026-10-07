#include "forge/cli_dashboard_args.h"

#include <cstdlib>
#include <string>

namespace forge {

bool is_dashboard_arg(const std::string& arg) {
    static const char* kDashboardFlags[] = {
        "--dashboard", "--dashboard-host", "--dashboard-ops", "--dashboard-port", "--no-open",
    };
    for (const char* f : kDashboardFlags) {
        if (arg == f)
            return true;
    }
    return false;
}

DashboardParseResult parse_dashboard_arg(const std::string& arg, int argc, char** argv, int& i,
                                         DashboardArgs& dash, std::string* err) {
    const auto need_value = [&]() -> const char* {
        if (i + 1 >= argc) {
            if (err)
                *err = "Error: " + arg + " requires an argument";
            return nullptr;
        }
        return argv[++i];
    };

    if (arg == "--dashboard") {
        dash.enabled = true;
        return DashboardParseResult::Ok;
    }

    if (arg == "--no-open") {
        dash.open_browser = false;
        return DashboardParseResult::Ok;
    }

    if (arg == "--dashboard-ops") {
        dash.capture_operators = true;
        // Naming the panel is an unambiguous request for the dashboard, same
        // reasoning as --dashboard-port.
        dash.enabled = true;
        return DashboardParseResult::Ok;
    }

    if (arg == "--dashboard-host") {
        const char* v = need_value();
        if (!v)
            return DashboardParseResult::Error;
        dash.host = v;
        return DashboardParseResult::Ok;
    }

    if (arg == "--dashboard-port") {
        const char* v = need_value();
        if (!v)
            return DashboardParseResult::Error;
        // strtol, not atoi: atoi silently returns 0 for "--dashboard-port abc",
        // and 0 here means "OS picks a free port" -- a typo would look like it
        // worked. The same class of bug the shared -ngl parser was written to
        // kill (see cli_common_args.h).
        char* end = nullptr;
        const long p = std::strtol(v, &end, 10);
        if (end == v || (end && *end != '\0') || p < 0 || p > 65535) {
            if (err)
                *err = "Error: --dashboard-port expects 0-65535 (0 = pick a free port), got: " +
                       std::string(v);
            return DashboardParseResult::Error;
        }
        dash.port = static_cast<int>(p);
        // Naming the port is an unambiguous request for the dashboard; requiring
        // "--dashboard --dashboard-port 9090" would just invite people to drop
        // the first half.
        dash.enabled = true;
        return DashboardParseResult::Ok;
    }

    return DashboardParseResult::NotMine;
}

}  // namespace forge