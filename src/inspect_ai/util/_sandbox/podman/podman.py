"""Podman sandbox environment with CRIU checkpoint/restore support.

Provides a sandbox environment backed by Podman containers and
podman-compose for orchestration. Supports CRIU checkpointing
with bridge networking (requires root).

Key differences from Docker sandbox:
- Uses `podman` and `podman-compose` commands
- Checkpoint/restore supports bridge networking with --tcp-established
- After restore, networks are reconnected to fix DNS resolution
- Containers must not use --init flag (breaks CRIU restore)
- Checkpoint requires root privileges
"""

import json
import os
import re
from logging import getLogger
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Union, overload

from typing_extensions import override

from inspect_ai._util.error import PrerequisiteError
from inspect_ai.util._subprocess import ExecResult, subprocess

from ..compose import COMPOSE_FILES, DOCKERFILE, ComposeConfig
from ..environment import (
    SandboxConnection,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
)
from ..limits import (
    OutputLimitExceededError,
    SandboxEnvironmentLimits,
)
from ..registry import sandboxenv

logger = getLogger(__name__)


@sandboxenv(name="podman")
class PodmanSandboxEnvironment(SandboxEnvironment):
    """Sandbox environment backed by Podman containers.

    Uses podman-compose for orchestration and supports CRIU
    checkpoint/restore with bridge networking.
    """

    @classmethod
    def config_files(cls) -> list[str]:
        return COMPOSE_FILES + [DOCKERFILE]

    @classmethod
    def is_docker_compatible(cls) -> bool:
        return True

    @classmethod
    def default_concurrency(cls) -> int | None:
        count = os.cpu_count() or 1
        return 2 * count

    @classmethod
    async def task_init(
        cls, task_name: str, config: SandboxEnvironmentConfigType | None
    ) -> None:
        await _validate_podman_prereqs()

    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        # Resolve compose file
        config_file = _resolve_config_file(config)
        if config_file is None:
            raise PrerequisiteError(
                "No compose file found for Podman sandbox. "
                "Provide a compose.yaml or Dockerfile."
            )

        # Generate a unique project name
        project_name = _project_name(task_name)

        # Build images
        await _podman_compose(
            ["build"], project_name=project_name, config_file=config_file
        )

        # Start services
        result = await _podman_compose(
            ["up", "-d"], project_name=project_name, config_file=config_file
        )
        if not result.success:
            raise RuntimeError(f"Failed to start Podman services: {result.stderr}")

        # List running containers to discover services
        ps_result = await _podman_compose(
            ["ps", "--format", "json"],
            project_name=project_name,
            config_file=config_file,
        )

        containers: list[dict[str, Any]] = []
        if ps_result.success and ps_result.stdout.strip():
            try:
                parsed = json.loads(ps_result.stdout)
                if isinstance(parsed, list):
                    containers = parsed
                else:
                    containers = [parsed]
            except json.JSONDecodeError:
                # podman-compose may not support --format json; fall back
                pass

        if not containers:
            # Fall back to podman ps with label filter
            ps_fallback = await subprocess(
                [
                    "podman",
                    "ps",
                    "--filter",
                    f"label=com.docker.compose.project={project_name}",
                    "--format",
                    "json",
                ],
                timeout=30,
            )
            if ps_fallback.success and ps_fallback.stdout.strip():
                try:
                    containers = json.loads(ps_fallback.stdout)
                    if not isinstance(containers, list):
                        containers = [containers]
                except json.JSONDecodeError:
                    pass

        if not containers:
            raise RuntimeError(f"No containers found for project '{project_name}'")

        # Create sandbox environments for each service
        environments: dict[str, SandboxEnvironment] = {}
        default_service: str | None = None

        for container in containers:
            # Extract service name from labels
            labels = container.get("Labels", {})
            if isinstance(labels, str):
                # Parse "key=value,key=value" format
                label_dict: dict[str, str] = {}
                for pair in labels.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        label_dict[k] = v
                labels = label_dict

            service = labels.get(
                "com.docker.compose.service",
                container.get("Service", ""),
            )
            container_name = container.get("Names", container.get("Name", ""))
            if isinstance(container_name, list):
                container_name = container_name[0] if container_name else ""

            if not service or not container_name:
                continue

            # Get working directory
            working_dir = await _container_working_dir(container_name)

            env = PodmanSandboxEnvironment(
                service=service,
                container_name=container_name,
                project_name=project_name,
                config_file=config_file,
                working_dir=working_dir,
            )
            environments[service] = env

        if not environments:
            raise RuntimeError(f"No services discovered for project '{project_name}'")

        # Ensure 'default' is first
        if "default" in environments:
            default_env = environments.pop("default")
            environments = {"default": default_env, **environments}
        elif default_service and default_service in environments:
            default_env = environments.pop(default_service)
            environments = {default_service: default_env, **environments}
        else:
            # Use the first service as default
            first_key = next(iter(environments))
            logger.warning(
                f"No 'default' service found; using '{first_key}' as default"
            )

        return environments

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        if not environments:
            return
        first_env = next(iter(environments.values()))
        if isinstance(first_env, PodmanSandboxEnvironment):
            await _podman_compose(
                ["down", "--volumes", "--remove-orphans"],
                project_name=first_env._project_name,
                config_file=first_env._config_file,
            )

    @classmethod
    async def task_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        cleanup: bool,
    ) -> None:
        pass

    def __init__(
        self,
        service: str,
        container_name: str,
        project_name: str,
        config_file: str,
        working_dir: str,
    ) -> None:
        super().__init__()
        self._service = service
        self._container_name = container_name
        self._project_name = project_name
        self._config_file = config_file
        self._working_dir = working_dir

    @override
    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        args: list[str] = ["podman", "exec"]

        # Working directory
        final_cwd = PurePosixPath(self._working_dir if cwd is None else cwd)
        if not final_cwd.is_absolute():
            final_cwd = PurePosixPath(self._working_dir) / final_cwd
        args.extend(["--workdir", str(final_cwd)])

        # Environment variables
        if env:
            for k, v in env.items():
                args.extend(["-e", f"{k}={v}"])

        # User
        if user:
            args.extend(["--user", user])

        # Interactive mode for input
        if input is not None:
            args.append("-i")

        args.append(self._container_name)
        args.extend(cmd)

        stdin: str | bytes | None = input
        return await subprocess(
            args,
            input=stdin,
            timeout=timeout or 120,
        )

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        target_path = self._resolve_path(file)

        # Ensure parent directory exists
        parent = str(PurePosixPath(target_path).parent)
        await subprocess(
            ["podman", "exec", self._container_name, "mkdir", "-p", parent],
            timeout=30,
        )

        # Write file via stdin
        data = contents if isinstance(contents, bytes) else contents.encode()
        result = await subprocess(
            [
                "podman",
                "exec",
                "-i",
                self._container_name,
                "sh",
                "-c",
                f"cat > '{target_path}'",
            ],
            input=data,
            timeout=60,
        )
        if not result.success:
            raise PermissionError(f"Failed to write file '{file}': {result.stderr}")

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> Union[str, bytes]:
        target_path = self._resolve_path(file)

        # Check file size first
        size_result = await subprocess(
            [
                "podman",
                "exec",
                self._container_name,
                "stat",
                "-c",
                "%s",
                target_path,
            ],
            timeout=30,
        )
        if not size_result.success:
            if "No such file" in size_result.stderr:
                raise FileNotFoundError(f"File not found: {file}")
            raise PermissionError(f"Failed to read file '{file}': {size_result.stderr}")

        file_size = int(size_result.stdout.strip())
        if file_size > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise OutputLimitExceededError(
                limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR,
                truncated_output=None,
            )

        # Read file contents
        result = await subprocess(
            ["podman", "exec", self._container_name, "cat", target_path],
            timeout=60,
        )
        if not result.success:
            raise PermissionError(f"Failed to read file '{file}': {result.stderr}")

        if text:
            return result.stdout
        else:
            return result.stdout.encode()

    @override
    async def connection(self, *, user: str | None = None) -> SandboxConnection:
        cmd = f"podman exec -it {self._container_name} /bin/bash"
        return SandboxConnection(
            type="podman",
            command=cmd,
            container=self._container_name,
        )

    @classmethod
    def supports_checkpoint(cls) -> bool:
        return True

    @override
    async def checkpoint_create(
        self,
        name: str,
        checkpoint_dir: str,
        leave_running: bool = True,
    ) -> bool:
        from .checkpoint import podman_checkpoint_create

        return await podman_checkpoint_create(
            container=self._container_name,
            name=name,
            checkpoint_dir=checkpoint_dir,
            leave_running=leave_running,
            tcp_established=True,
        )

    @override
    async def checkpoint_restore(
        self,
        name: str,
        checkpoint_dir: str,
    ) -> bool:
        from .checkpoint import podman_checkpoint_restore

        return await podman_checkpoint_restore(
            container=self._container_name,
            name=name,
            checkpoint_dir=checkpoint_dir,
            tcp_established=True,
        )

    @override
    async def checkpoint_delete(
        self,
        name: str,
        checkpoint_dir: str,
    ) -> None:
        from .checkpoint import podman_checkpoint_delete

        await podman_checkpoint_delete(
            container=self._container_name,
            name=name,
            checkpoint_dir=checkpoint_dir,
        )

    def default_polling_interval(self) -> float:
        return 0.2

    def _resolve_path(self, file: str) -> str:
        """Resolve a file path relative to the working directory."""
        path = PurePosixPath(file)
        if not path.is_absolute():
            path = PurePosixPath(self._working_dir) / path
        return str(path)


# ---- Internal helper functions ----


async def _validate_podman_prereqs() -> None:
    """Validate that Podman and podman-compose are available."""
    podman_result = await subprocess(["podman", "--version"], timeout=10)
    if not podman_result.success:
        raise PrerequisiteError(
            "Podman is not installed or not in PATH. "
            "Install Podman: https://podman.io/getting-started/installation"
        )

    compose_result = await subprocess(["podman-compose", "--version"], timeout=10)
    if not compose_result.success:
        raise PrerequisiteError(
            "podman-compose is not installed or not in PATH. "
            "Install: pip install podman-compose"
        )


def _resolve_config_file(
    config: SandboxEnvironmentConfigType | None,
) -> str | None:
    """Resolve a compose file path from config."""
    if isinstance(config, str):
        path = Path(config).resolve()
        if path.exists():
            return str(path)
        return None
    elif isinstance(config, ComposeConfig):
        # Write ComposeConfig to a temporary file
        import tempfile

        import yaml

        data = config.model_dump(mode="json", by_alias=True, exclude_none=True)
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="podman-compose-")
        with os.fdopen(fd, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return path
    else:
        # Look for standard compose files in CWD
        for name in COMPOSE_FILES:
            if Path(name).exists():
                return str(Path(name).resolve())
        return None


def _project_name(task_name: str) -> str:
    """Generate a unique project name for Podman compose."""
    from shortuuid import uuid

    task = task_name.lower()
    task = re.sub(r"[^a-z\d\-_]", "-", task)
    task = re.sub(r"-+", "-", task)
    if not task:
        task = "task"
    return f"inspect-podman-{task[:12].rstrip('_')}-{uuid().lower()[:6]}"


async def _podman_compose(
    args: list[str],
    project_name: str,
    config_file: str,
) -> ExecResult[str]:
    """Run a podman-compose command."""
    cmd = [
        "podman-compose",
        "-p",
        project_name,
        "-f",
        config_file,
        *args,
    ]
    return await subprocess(cmd, timeout=300)


async def _container_working_dir(container_name: str, default: str = "/") -> str:
    """Get the working directory of a container."""
    result = await subprocess(
        ["podman", "exec", container_name, "sh", "-c", "pwd"],
        timeout=30,
    )
    if result.success:
        return result.stdout.strip()
    return default
