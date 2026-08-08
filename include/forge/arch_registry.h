#pragma once

// All types (ConfigParserRegistry, WeightInitRegistry, ArchCapabilityRegistry,
// and their auto-register structs and macros) are defined in model.h.
// This header is kept for backward compatibility.
#include "model.h"

namespace forge {

// Defined in arch_registrations.cpp. Referencing this symbol forces the linker
// to include arch_registrations.o, ensuring static auto-registration objects
// (EngineAutoRegister, ConfigParserAutoRegister, etc.) are constructed.
extern volatile bool _arch_registrations_linked;

// Call from any binary that links forge_model directly (e.g. forge-inspect) so
// the linker pulls in arch_registrations.o and its static auto-registration
// objects. No-op at runtime.
void force_link_arch_registrations();

}  // namespace forge
