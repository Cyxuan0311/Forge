# Stage 1: Build
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ARG USE_CUBLAS=ON
ARG USE_OPENBLAS=OFF

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    python3-dev \
    python3-pip \
    python3-numpy \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# FORGE_BUILD_DASHBOARD is intentionally OFF in the image.
#
# The builder is Ubuntu 22.04, whose `apt install nodejs npm` yields Node v12,
# and Vite 7 requires ^20.19 || >=22.12 -- so the npm frontend cannot be built
# here without adding the NodeSource repository (another network dependency and
# another thing to keep patched). The dashboard is a developer tool; the
# container ships the inference engine. Note the consequence: /api/metrics still
# answers, and /dashboard/ explains how to rebuild with the frontend.
#
# To enable it in the image, install Node 20+ and add:
#   -DFORGE_BUILD_DASHBOARD=ON
RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75;86;89" \
    -DUSE_CUBLAS=${USE_CUBLAS} \
    -DUSE_OPENBLAS=${USE_OPENBLAS} \
    -DFORGE_BUILD_DASHBOARD=OFF \
    && cmake --build build -j $(nproc)

# Stage 2: Runtime
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp5 \
    python3 \
    python3-pip \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/build/forge-cli /usr/local/bin/forge-cli
COPY --from=builder /app/build/forge*.so /usr/local/lib/python3.10/dist-packages/
COPY --from=builder /app/examples /app/examples

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages:$LD_LIBRARY_PATH

ENTRYPOINT ["forge-cli"]
CMD ["--help"]
