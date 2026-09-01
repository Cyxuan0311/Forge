import sys
import numpy as np
sys.path.insert(0, "build")
import forge

MODEL = "/mnt/g/AI/MiMo-7B-RL-Q4_K_M/MiMo-7B-RL-Q4_K_M.gguf"
for dt in ["fp32", "fp8_e4m3", "fp8_e5m2", "q8_0", "f16"]:
    m = forge.Model()
    m.load_auto(MODEL, device="cuda")
    c = m.create_context(kv_cache_dtype=dt, gpu_layers=-1)
    c.forward(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    st = c.memory_stats()
    print(f"requested={dt:9s} -> kv_cache_dtype={st.get('kv_cache_dtype')} "
          f"type_k={st.get('kv_cache_type_k')} type_v={st.get('kv_cache_type_v')} "
          f"active_bytes={st.get('kv_cache_active_bytes')}")
