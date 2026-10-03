# MI210 Docker access retry

Current retry confirmed through the pre-authorized docker inspect command:

- int8-vllm-mi210-ptqr is running, not restarting, not OOM-killed.
- Image: sha256:8df86c75bfb56d3654bca7dee59838c323b6250eb7649a0f28de9c13c930d79a.
- Container has /dev/kfd and /dev/dri assigned, host networking, and /data/models mounted at /models.
- CDNA2 decode is enabled, KV tile64, UA tile32 and MAXQ16; AITER enabled.
- Target quantization: bits8/group_size128, all eight indexed weight shards exist.
- Draft quantization: bits8/group_size128, its indexed weight shard exists.
- Four split-selection CPU tests pass again.

Execution remains blocked in this session: docker exec and docker ps return permission denied on the Docker socket; GPU device paths are not exposed in the session; the service health endpoint at 10.168.1.4:8080 cannot be connected to. Container running status alone does not establish healthy inference or GPU numerical correctness. No new 48K throughput measurements, split sweep, service restart or Docker archive was performed. Existing median remains 84.891 tok/s; 90–100 tok/s is unverified.

Next required capability is authorized Docker execution on this host or an existing authorized MI210 remote execution environment. Experiment sources and qualified archive tool are already on this branch; GPU numeric, graph replay and whole-model gates still precede promotion/export.
