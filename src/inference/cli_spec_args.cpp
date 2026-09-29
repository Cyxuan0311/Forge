#include "forge/cli_spec_args.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace forge {

bool is_spec_arg(const std::string& arg) {
    static const char* kSpecFlags[] = {
        "--no-adaptive-draft",
        "--no-ngram",
        "--spec",
        "--spec-draft",
        "--spec-draft-arch",
        "--spec-draft-mask-id",
        "--spec-draft-n",
        "--spec-draft-ngl",
        "--spec-draft-target-layers",
        "--spec-dspark-conf",
        "--spec-dspark-topk",
        "--spec-min-acc",
        "--spec-model",
        "--spec-mtp",
        "--spec-n-min",
        "--spec-ngram-min",
        "--spec-ngram-mod",
        "--spec-ngram-mod-max",
        "--spec-ngram-mod-min",
        "--spec-ngram-mod-n",
        "--spec-ngram-n",
        "--spec-p-min",
        "--spec-stats",
    };
    for (const char* f : kSpecFlags) {
        if (arg == f) return true;
    }
    return false;
}

SpecParseResult parse_spec_arg(const std::string& arg, int argc, char** argv, int& i,
                               SpeculativeConfig& spec, std::string* err) {
    const auto need_value = [&]() -> const char* {
        if (i + 1 >= argc) {
            if (err) *err = "Error: " + arg + " requires an argument";
            return nullptr;
        }
        return argv[++i];
    };

    if (arg == "--spec") {
        spec.enabled = true;
        spec.use_ngram = true;
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-mtp") {
        spec.enabled = true;
        spec.use_mtp = true;
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-draft") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.enabled = true;
        spec.use_ngram = true;
        spec.n_draft = std::atoi(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-n-min") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.enabled = true;
        spec.use_ngram = true;
        spec.n_min = std::atoi(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-ngram-n") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.enabled = true;
        spec.use_ngram = true;
        spec.ngram_n = std::atoi(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-ngram-min") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.enabled = true;
        spec.use_ngram = true;
        spec.ngram_min = std::atoi(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-p-min") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.enabled = true;
        spec.p_min = std::atof(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-model") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.enabled = true;
        spec.draft_model_path = argv[i];
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-draft-ngl") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.draft_gpu_layers = std::atoi(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-draft-arch") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;

        const std::string arch = v;
        if (arch != "dflash" && arch != "dspark") {
            if (err) *err = "Error: --spec-draft-arch must be 'dflash' or 'dspark'";
            return SpecParseResult::Error;
        }
        spec.enabled = true;
        spec.draft_arch = arch;
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-draft-target-layers") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.draft_target_layers.clear();
        const std::string list = v;
        size_t start = 0;
        while (true) {
            const size_t comma = list.find(',', start);
            const std::string piece =
                list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!piece.empty()) {
                try {
                    spec.draft_target_layers.push_back(std::stoi(piece));
                } catch (const std::exception&) {
                    if (err)
                        *err =
                            "Error: --spec-draft-target-layers expects comma-separated layer "
                            "indices";
                    return SpecParseResult::Error;
                }
            }
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        if (spec.draft_target_layers.empty()) {
            if (err) *err = "Error: --spec-draft-target-layers is empty";
            return SpecParseResult::Error;
        }
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-draft-mask-id") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.draft_mask_token_id = std::atoi(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-draft-n") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.enabled = true;
        spec.draft_n_spec = std::atoi(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-dspark-topk") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.enabled = true;
        spec.dspark_top_k = std::atoi(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-dspark-conf") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.enabled = true;
        spec.dspark_conf_threshold = std::atof(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-min-acc") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.enabled = true;
        spec.min_accept_rate = std::atof(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-ngram-mod") {
        spec.enabled = true;
        spec.use_ngram_mod = true;
        return SpecParseResult::Ok;
    }

    if (arg == "--no-ngram") {
        spec.enabled = true;
        spec.no_ngram = true;
        spec.use_ngram = false;
        return SpecParseResult::Ok;
    }

    if (arg == "--no-adaptive-draft") {
        spec.enabled = true;
        spec.adaptive_draft = false;
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-ngram-mod-n") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.enabled = true;
        spec.use_ngram_mod = true;
        spec.ngram_mod_n = std::atoi(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-ngram-mod-min") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.enabled = true;
        spec.use_ngram_mod = true;
        spec.ngram_mod_n_min = std::atoi(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-ngram-mod-max") {
        const char* v = need_value();
        if (!v) return SpecParseResult::Error;
        spec.enabled = true;
        spec.use_ngram_mod = true;
        spec.ngram_mod_n_max = std::atoi(v);
        return SpecParseResult::Ok;
    }

    if (arg == "--spec-stats") {
        spec.print_stats = true;
        return SpecParseResult::Ok;
    }

    return SpecParseResult::NotMine;
}

}  // namespace forge
