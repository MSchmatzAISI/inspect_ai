# Podman Checkpoint/Restore Notes

## Spike Results (2026-03-04)

Tested with Podman 4.9.3, CRIU 4.2, runc, Ubuntu 24.04.

### What works
- `podman container checkpoint --leave-running` with bridge networking
- `--tcp-established` flag for checkpointing active TCP connections
- In-place restore (`podman stop` then `podman container restore`)
- Process state preservation (Python server in-memory counter survived restore)
- IP-based connectivity works immediately after restore

### What doesn't work
- **`init: true` in compose** — catatonit bind mount breaks CRIU restore
- **`--export`/`--import` restore** — CRIU mount namespace error
- **DNS after restore** — aardvark-dns loses track of restored containers
- **Rootless mode** — checkpoint requires root

### DNS fix
After restore, DNS resolution breaks because aardvark-dns doesn't know about
the restored container. Fixed by disconnecting and reconnecting each network:
```
podman network disconnect <network> <container>
podman network connect <network> <container>
```
This is handled automatically in `checkpoint.py:_reconnect_networks()`.

## `init: true` incompatibility

`init: true` tells the container runtime to inject a tiny init process
(catatonit for Podman, docker-init/tini for Docker) as PID 1 via a bind
mount from the host. This init process:
- Reaps zombie child processes
- Forwards signals (SIGTERM/SIGINT) to children for graceful shutdown

CRIU restore fails because it tries to recreate the bind mount and gets:
```
Can't remount /tmp/.criu.mntns.../mnt-... with MS_PRIVATE: Invalid argument
```

### Workaround
Install an init process inside the image instead of using runtime injection:

```dockerfile
RUN apt-get update && apt-get install -y tini
ENTRYPOINT ["tini", "--"]
CMD ["tail", "-f", "/dev/null"]
```

Then remove `init: true` from `compose.yaml`. Same zombie-reaping and
signal-forwarding behavior, no bind mount for CRIU to choke on.

## Requirements for checkpoint/restore with bridge networking

1. Podman must run as root (CRIU requirement)
2. Containers must NOT use `init: true` (use in-image tini instead)
3. Use in-place checkpoint/restore (not export/import)
4. After restore, networks are automatically reconnected for DNS
