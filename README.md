# llamacpp-cuda133-staging

Throwaway staging repo to validate the CUDA 13.3 build leg of the Unsloth
llama.cpp prebuilt workflow (unslothai/llama.cpp PR #17, head oobabooga:unsloth-build).

It contains only the parent workflow (trimmed to resolve + the Linux CUDA leg),
the reusable CUDA child workflow, and the bundle packager. The build checks out
ggml-org/llama.cpp source separately, so the upstream tree is not vendored here.

Run with workflow_dispatch: tag=b9518, only_profile=cuda13-newer, publish=false.
This produces a single app-b9518-linux-x64-cuda13-newer.tar.gz artifact built
with CUDA toolkit 13.3 (pulled from NVIDIA's redist CDN), for compatibility
testing on a Blackwell host. Delete after use.
