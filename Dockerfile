# Stage 1: Build
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ARG USE_CUBLAS=ON
ARG USE_OPENBLAS=OFF

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    python3-dev \
    python3-pip \
    python3-numpy \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75;86;89" \
    -DUSE_CUBLAS=${USE_CUBLAS} \
    -DUSE_OPENBLAS=${USE_OPENBLAS} \
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

COPY --from=builder /app/build/nanoinfer-cli /usr/local/bin/nanoinfer-cli
COPY --from=builder /app/build/nanoinfer*.so /usr/local/lib/python3.10/dist-packages/
COPY --from=builder /app/examples /app/examples

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages:$LD_LIBRARY_PATH

ENTRYPOINT ["nanoinfer-cli"]
CMD ["--help"]
