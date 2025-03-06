import os
import re
from logging import getLogger
from typing import Any, Callable, Generator, Literal

import anyio
from pydantic import BaseModel
from pydantic_core import to_json

from inspect_ai._util._async import current_async_backend
from inspect_ai._util.constants import ALL_LOG_FORMATS, EVAL_LOG_FORMAT
from inspect_ai._util.file import (
    FileInfo,
    file,
    filesystem,
)
from inspect_ai._util.json import jsonable_python
from inspect_ai.log._condense import resolve_sample_attachments

from ._log import EvalLog, EvalSample
from ._recorders import recorder_type_for_format, recorder_type_for_location

logger = getLogger(__name__)


class EvalLogInfo(BaseModel):
    """File info and task identifiers for eval log."""

    name: str
    """Name of file."""

    type: str
    """Type of file (file or directory)"""

    size: int
    """File size in bytes."""

    mtime: float | None
    """File modification time (None if the file is a directory on S3)."""

    task: str
    """Task name."""

    task_id: str
    """Task id."""

    suffix: str | None
    """Log file suffix (e.g. "-scored")"""


def list_eval_logs(
    log_dir: str = os.environ.get("INSPECT_LOG_DIR", "./logs"),
    formats: list[Literal["eval", "json"]] | None = None,
    filter: Callable[[EvalLog], bool] | None = None,
    recursive: bool = True,
    descending: bool = True,
    fs_options: dict[str, Any] = {},
) -> list[EvalLogInfo]:
    """List all eval logs in a directory.

    Args:
      log_dir (str): Log directory (defaults to INSPECT_LOG_DIR)
      formats (Literal["eval", "json"]): Formats to list (default
        to listing all formats)
      filter (Callable[[EvalLog], bool]): Filter to limit logs returned.
         Note that the EvalLog instance passed to the filter has only
         the EvalLog header (i.e. does not have the samples or logging output).
      recursive (bool): List log files recursively (defaults to True).
      descending (bool): List in descending order.
      fs_options (dict[str, Any]): Optional. Additional arguments to pass through
          to the filesystem provider (e.g. `S3FileSystem`).

    Returns:
       List of EvalLog Info.

    """
    # get the eval logs
    fs = filesystem(log_dir, fs_options)
    if fs.exists(log_dir):
        logger.debug(f"Listing eval logs for {log_dir}")
        eval_logs = log_files_from_ls(
            fs.ls(log_dir, recursive=recursive), formats, descending
        )
        logger.debug(f"Listing eval logs for {log_dir} completed")
    else:
        return []

    # apply filter if requested
    if filter:
        return [
            log
            for log in eval_logs
            if filter(read_eval_log(log.name, header_only=True))
        ]
    else:
        return eval_logs


def write_eval_log(
    log: EvalLog,
    location: str | FileInfo | None = None,
    format: Literal["eval", "json", "auto"] = "auto",
) -> None:
    """Write an evaluation log.

    Args:
       log (EvalLog): Evaluation log to write.
       location (str | FileInfo): Location to write log to.
       format (Literal["eval", "json", "auto"]): Write to format
          (defaults to 'auto' based on `log_file` extension)
    """
    # don't mix trio and asyncio
    if current_async_backend() == "trio":
        raise RuntimeError(
            "write_eval_log cannot be called from a trio async context (please use write_eval_log_async instead)"
        )

    # will use s3fs and is not called from main inspect solver/scorer/tool/sandbox
    # flow, so force the use of asyncio
    anyio.run(write_eval_log_async, log, location, format, backend="asyncio")


async def write_eval_log_async(
    log: EvalLog,
    location: str | FileInfo | None = None,
    format: Literal["eval", "json", "auto"] = "auto",
) -> None:
    """Write an evaluation log.

    Args:
       log (EvalLog): Evaluation log to write.
       location (str | FileInfo): Location to write log to.
       format (Literal["eval", "json", "auto"]): Write to format
          (defaults to 'auto' based on `log_file` extension)
    """
    # resolve location
    if location is None:
        if log.location:
            location = log.location
        else:
            raise ValueError(
                "EvalLog passe to write_eval_log does not have a location, so you must pass an explicit location"
            )
    location = location if isinstance(location, str) else location.name

    logger.debug(f"Writing eval log to {location}")

    # get recorder type
    if format == "auto":
        recorder_type = recorder_type_for_location(location)
    else:
        recorder_type = recorder_type_for_format(format)
    await recorder_type.write_log(location, log)

    logger.debug(f"Writing eval log to {location} completed")


def write_log_dir_manifest(
    log_dir: str,
    *,
    filename: str = "logs.json",
    output_dir: str | None = None,
    fs_options: dict[str, Any] = {},
) -> None:
    """Write a manifest for a log directory.

    A log directory manifest is a dictionary of EvalLog headers (EvalLog w/o samples)
    keyed by log file names (names are relative to the log directory)

    Args:
      log_dir (str): Log directory to write manifest for.
      filename (str): Manifest filename (defaults to "logs.json")
      output_dir (str | None): Output directory for manifest (defaults to log_dir)
      fs_options (dict[str,Any]): Optional. Additional arguments to pass through
        to the filesystem provider (e.g. `S3FileSystem`).
    """
    # resolve log dir to full path
    fs = filesystem(log_dir)
    log_dir = fs.info(log_dir).name

    # list eval logs
    logs = list_eval_logs(log_dir)

    # resolve to manifest (make filenames relative to the log dir)
    names = [manifest_eval_log_name(log, log_dir, fs.sep) for log in logs]
    headers = read_eval_log_headers(logs)
    manifest_logs = dict(zip(names, headers))

    # form target path and write
    output_dir = output_dir or log_dir
    fs = filesystem(output_dir)
    manifest = f"{output_dir}{fs.sep}{filename}"
    manifest_json = to_json(
        value=manifest_logs, indent=2, exclude_none=True, fallback=lambda _x: None
    )
    with file(manifest, mode="wb", fs_options=fs_options) as f:
        f.write(manifest_json)


def read_eval_log(
    log_file: str | EvalLogInfo,
    header_only: bool = False,
    resolve_attachments: bool = False,
    format: Literal["eval", "json", "auto"] = "auto",
) -> EvalLog:
    """Read an evaluation log.

    Args:
       log_file (str | FileInfo): Log file to read.
       header_only (bool): Read only the header (i.e. exclude
          the "samples" and "logging" fields). Defaults to False.
       resolve_attachments (bool): Resolve attachments (e.g. images)
          to their full content.
       format (Literal["eval", "json", "auto"]): Read from format
          (defaults to 'auto' based on `log_file` extension)

    Returns:
       EvalLog object read from file.
    """
    # don't mix trio and asyncio
    if current_async_backend() == "trio":
        raise RuntimeError(
            "read_eval_log cannot be called from a trio async context (please use read_eval_log_async instead)"
        )

    # will use s3fs and is not called from main inspect solver/scorer/tool/sandbox
    # flow, so force the use of asyncio
    return anyio.run(
        read_eval_log_async,
        log_file,
        header_only,
        resolve_attachments,
        format,
        backend="asyncio",
    )


async def read_eval_log_async(
    log_file: str | EvalLogInfo,
    header_only: bool = False,
    resolve_attachments: bool = False,
    format: Literal["eval", "json", "auto"] = "auto",
) -> EvalLog:
    """Read an evaluation log.

    Args:
       log_file (str | FileInfo): Log file to read.
       header_only (bool): Read only the header (i.e. exclude
          the "samples" and "logging" fields). Defaults to False.
       resolve_attachments (bool): Resolve attachments (e.g. images)
          to their full content.
       format (Literal["eval", "json", "auto"]): Read from format
          (defaults to 'auto' based on `log_file` extension)

    Returns:
       EvalLog object read from file.
    """
    # resolve to file path
    log_file = log_file if isinstance(log_file, str) else log_file.name
    logger.debug(f"Reading eval log from {log_file}")

    # get recorder type
    if format == "auto":
        recorder_type = recorder_type_for_location(log_file)
    else:
        recorder_type = recorder_type_for_format(format)
    log = await recorder_type.read_log(log_file, header_only)

    # resolve attachement if requested
    if resolve_attachments and log.samples:
        log.samples = [resolve_sample_attachments(sample) for sample in log.samples]

    # provide sample ids if they aren't there
    if log.eval.dataset.sample_ids is None and log.samples is not None:
        sample_ids: dict[str | int, None] = {}
        for sample in log.samples:
            if sample.id not in sample_ids:
                sample_ids[sample.id] = None
        log.eval.dataset.sample_ids = list(sample_ids.keys())

    logger.debug(f"Completed reading eval log from {log_file}")

    return log


def read_eval_log_headers(
    log_files: list[str] | list[EvalLogInfo],
) -> list[EvalLog]:
    # will use s3fs and is not called from main inspect solver/scorer/tool/sandbox
    # flow, so force the use of asyncio
    return anyio.run(read_eval_log_headers_async, log_files, backend="asyncio")


async def read_eval_log_headers_async(
    log_files: list[str] | list[EvalLogInfo],
) -> list[EvalLog]:
    return [
        await read_eval_log_async(log_file, header_only=True) for log_file in log_files
    ]


def read_eval_log_sample(
    log_file: str | EvalLogInfo,
    id: int | str,
    epoch: int = 1,
    resolve_attachments: bool = False,
    format: Literal["eval", "json", "auto"] = "auto",
) -> EvalSample:
    """Read a sample from an evaluation log.

    Args:
       log_file (str | FileInfo): Log file to read.
       id (int | str): Sample id to read.
       epoch (int): Epoch for sample id (defaults to 1)
       resolve_attachments (bool): Resolve attachments (e.g. images)
          to their full content.
       format (Literal["eval", "json", "auto"]): Read from format
          (defaults to 'auto' based on `log_file` extension)

    Returns:
       EvalSample object read from file.

    Raises:
       IndexError: If the passed id and epoch are not found.
    """
    # don't mix trio and asyncio
    if current_async_backend() == "trio":
        raise RuntimeError(
            "read_eval_log_sample cannot be called from a trio async context (please use read_eval_log_sample_async instead)"
        )

    # will use s3fs and is not called from main inspect solver/scorer/tool/sandbox
    # flow, so force the use of asyncio
    return anyio.run(
        read_eval_log_sample_async,
        log_file,
        id,
        epoch,
        resolve_attachments,
        format,
        backend="asyncio",
    )


async def read_eval_log_sample_async(
    log_file: str | EvalLogInfo,
    id: int | str,
    epoch: int = 1,
    resolve_attachments: bool = False,
    format: Literal["eval", "json", "auto"] = "auto",
) -> EvalSample:
    """Read a sample from an evaluation log.

    Args:
       log_file (str | FileInfo): Log file to read.
       id (int | str): Sample id to read.
       epoch (int): Epoch for sample id (defaults to 1)
       resolve_attachments (bool): Resolve attachments (e.g. images)
          to their full content.
       format (Literal["eval", "json", "auto"]): Read from format
          (defaults to 'auto' based on `log_file` extension)

    Returns:
       EvalSample object read from file.

    Raises:
       IndexError: If the passed id and epoch are not found.
    """
    # resolve to file path
    log_file = log_file if isinstance(log_file, str) else log_file.name

    if format == "auto":
        recorder_type = recorder_type_for_location(log_file)
    else:
        recorder_type = recorder_type_for_format(format)
    sample = await recorder_type.read_log_sample(log_file, id, epoch)

    if resolve_attachments:
        sample = resolve_sample_attachments(sample)

    return sample


def read_eval_log_samples(
    log_file: str | EvalLogInfo,
    all_samples_required: bool = True,
    resolve_attachments: bool = False,
    format: Literal["eval", "json", "auto"] = "auto",
) -> Generator[EvalSample, None, None]:
    """Read all samples from an evaluation log incrementally.

    Generator for samples in a log file. Only one sample at a time
    will be read into memory and yielded to the caller.

    Args:
       log_file (str | FileInfo): Log file to read.
       all_samples_required (bool): All samples must be included in
          the file or an IndexError is thrown.
       resolve_attachments (bool): Resolve attachments (e.g. images)
          to their full content.
       format (Literal["eval", "json", "auto"]): Read from format
          (defaults to 'auto' based on `log_file` extension)

    Returns:
       Generator of EvalSample objects in the log file.

    Raises:
       IndexError: If `all_samples_required` is `True` and one of the target
          samples does not exist in the log file.
    """
    # read header
    log_header = read_eval_log(log_file, header_only=True, format=format)

    # do we have the list of samples?
    if log_header.eval.dataset.sample_ids is None:
        raise RuntimeError(
            "This log file does not include sample_ids "
            + "(fully reading and re-writing the log will add sample_ids)"
        )

    # if the status is not success and all_samples_required, this is an error
    if log_header.status != "success" and all_samples_required:
        raise RuntimeError(
            f"This log does not have all samples (status={log_header.status}). "
            + "Specify all_samples_required=False to read the samples that exist."
        )

    # loop over samples and epochs
    for sample_id in log_header.eval.dataset.sample_ids:
        for epoch_id in range(1, (log_header.eval.config.epochs or 1) + 1):
            try:
                sample = read_eval_log_sample(
                    log_file=log_file,
                    id=sample_id,
                    epoch=epoch_id,
                    resolve_attachments=resolve_attachments,
                    format=format,
                )
                yield sample
            except IndexError:
                if all_samples_required:
                    raise


def manifest_eval_log_name(info: EvalLogInfo, log_dir: str, sep: str) -> str:
    # ensure that log dir has a trailing seperator
    if not log_dir.endswith(sep):
        log_dir = f"{log_dir}/"

    # slice off log_dir from the front
    log = info.name.replace(log_dir, "")

    # manifests are web artifacts so always use forward slash
    return log.replace("\\", "/")


def log_files_from_ls(
    ls: list[FileInfo],
    formats: list[Literal["eval", "json"]] | None,
    descending: bool = True,
) -> list[EvalLogInfo]:
    extensions = [f".{format}" for format in (formats or ALL_LOG_FORMATS)]
    return [
        log_file_info(file)
        for file in sorted(
            ls, key=lambda file: (file.mtime if file.mtime else 0), reverse=descending
        )
        if file.type == "file" and is_log_file(file.name, extensions)
    ]


log_file_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}[:-]\d{2}[:-]\d{2}.*$"


def is_log_file(file: str, extensions: list[str]) -> bool:
    parts = file.replace("\\", "/").split("/")
    name = parts[-1]

    if name.endswith(f".{EVAL_LOG_FORMAT}"):
        return True
    else:
        return re.match(log_file_pattern, name) is not None and any(
            [name.endswith(suffix) for suffix in extensions]
        )


def log_file_info(info: FileInfo) -> "EvalLogInfo":
    # extract the basename and split into parts
    # (deal with previous logs had the model in their name)
    basename = os.path.splitext(info.name)[0]
    parts = basename.split("/").pop().split("_")
    if len(parts) == 1:
        task = ""
        task_id = ""
        suffix = None
    elif len(parts) == 2:
        task = parts[1]
        task_id = ""
        suffix = None
    else:
        last_idx = 3 if len(parts) > 3 else 2
        task = parts[1]
        part3 = parts[last_idx].split("-")
        task_id = part3[0]
        suffix = task_id[2] if len(part3) > 1 else None
    return EvalLogInfo(
        name=info.name,
        type=info.type,
        size=info.size,
        mtime=info.mtime,
        task=task,
        task_id=task_id,
        suffix=suffix,
    )


def eval_log_json(log: EvalLog) -> bytes:
    # serialize to json (ignore values that are unserializable)
    # these values often result from solvers using metadata to
    # pass around 'live' objects -- this is fine to do and we
    # don't want to prevent it at the serialization level
    return to_json(
        value=jsonable_python(log),
        indent=2,
        exclude_none=True,
        fallback=lambda _x: None,
    )


def eval_log_json_str(log: EvalLog) -> str:
    return eval_log_json(log).decode()
