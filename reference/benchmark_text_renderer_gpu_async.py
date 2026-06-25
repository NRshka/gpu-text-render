#!/usr/bin/env python3
import argparse
import asyncio
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc.aio import InferenceServerClient


DEFAULT_URL = os.environ.get("TRITON_TEXT_RENDERER_URL", "localhost:8001")
DEFAULT_MODEL_NAME = "text_renderer_gpu"
IMAGE_HEIGHT = 1200
IMAGE_WIDTH = 900
IMAGE_CHANNELS = 3
COMMAND_ABI_VERSION = 2


@dataclass(frozen=True)
class RequestPayload:
    image: np.ndarray
    command_version: np.ndarray
    command_bytes: np.ndarray
    batch_bytes: np.ndarray
    label: str


@dataclass
class BenchmarkStats:
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_latency_s: float = 0.0
    latencies_s: list[float] | None = None
    output_bytes: int = 0
    first_error: str = ""

    def __post_init__(self) -> None:
        if self.latencies_s is None:
            self.latencies_s = []


class TritonGpuBenchmarkClient:
    def __init__(self, url: str, model_name: str, timeout_s: float | None) -> None:
        self._url = url
        self._model_name = model_name
        self._timeout_s = timeout_s
        self._client: InferenceServerClient | None = None

    def _get_client(self) -> InferenceServerClient:
        if self._client is None:
            self._client = InferenceServerClient(url=self._url)
        return self._client

    async def is_ready(self) -> bool:
        try:
            client = self._get_client()
            return await client.is_server_live()
        except Exception:
            return False

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def infer(self, payload: RequestPayload) -> np.ndarray:
        request_image = np.expand_dims(payload.image, axis=0)
        request_command_version = np.expand_dims(payload.command_version, axis=0)
        request_command_bytes = np.expand_dims(payload.command_bytes, axis=0)
        request_batch_bytes = np.expand_dims(payload.batch_bytes, axis=0)

        inputs = [
            grpcclient.InferInput("input_images", request_image.shape, "UINT8"),
            grpcclient.InferInput(
                "command_version", request_command_version.shape, "INT32"
            ),
            grpcclient.InferInput("command_bytes", request_command_bytes.shape, "UINT8"),
            grpcclient.InferInput("batch_bytes", request_batch_bytes.shape, "UINT8"),
        ]
        inputs[0].set_data_from_numpy(request_image)
        inputs[1].set_data_from_numpy(request_command_version)
        inputs[2].set_data_from_numpy(request_command_bytes)
        inputs[3].set_data_from_numpy(request_batch_bytes)

        client = self._get_client()
        result = await client.infer(
            model_name=self._model_name,
            inputs=inputs,
            client_timeout=self._timeout_s,
        )
        output = result.as_numpy("rendered_image")
        if output is None:
            raise RuntimeError("Triton response does not contain rendered_image")
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Async throughput benchmark for the Triton text_renderer_gpu model. "
            "Use either real request payloads from disk or synthetic empty-command inputs."
        )
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Triton gRPC URL")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Triton model name to benchmark",
    )
    parser.add_argument(
        "--request-npz",
        action="append",
        default=[],
        help=(
            "Path to a .npz payload containing input_images, command_version, "
            "command_bytes, and batch_bytes. May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--request-dir",
        action="append",
        default=[],
        help=(
            "Path to a directory with image.npy, command_bytes.bin, batch_bytes.bin, "
            "and optional command_version.txt or command_version.npy. May be repeated."
        ),
    )
    parser.add_argument(
        "--synthetic-requests",
        type=int,
        default=1,
        help=(
            "Number of synthetic request variants to build when no request files are provided"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent in-flight requests",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Measured benchmark duration in seconds",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=2.0,
        help="Warmup duration in seconds",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Optional hard cap on measured requests, 0 means unlimited",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request Triton client timeout in seconds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for synthetic request generation",
    )
    parser.add_argument(
        "--synthetic-mode",
        choices=("zeros", "noise", "gradient"),
        default="gradient",
        help="Synthetic image pattern to use when no payloads are supplied",
    )
    parser.add_argument(
        "--save-sample-npz",
        default="",
        help=(
            "Optional output path for saving the first synthetic payload as a .npz template"
        ),
    )
    return parser.parse_args()


def load_payload_from_npz(path: Path) -> RequestPayload:
    with np.load(path, allow_pickle=False) as data:
        image = np.asarray(data["input_images"], dtype=np.uint8)
        command_version = np.asarray(data["command_version"], dtype=np.int32)
        command_bytes = np.asarray(data["command_bytes"], dtype=np.uint8)
        batch_bytes = np.asarray(data["batch_bytes"], dtype=np.uint8)
    return validate_payload(
        RequestPayload(
            image=image,
            command_version=command_version,
            command_bytes=command_bytes,
            batch_bytes=batch_bytes,
            label=path.name,
        )
    )


def load_payload_from_dir(path: Path) -> RequestPayload:
    image = np.load(path / "image.npy")

    command_version_path_npy = path / "command_version.npy"
    command_version_path_txt = path / "command_version.txt"
    if command_version_path_npy.exists():
        command_version = np.asarray(np.load(command_version_path_npy), dtype=np.int32)
    elif command_version_path_txt.exists():
        command_version = np.asarray(
            [int(command_version_path_txt.read_text(encoding="utf-8").strip())],
            dtype=np.int32,
        )
    else:
        command_version = np.asarray([COMMAND_ABI_VERSION], dtype=np.int32)

    command_bytes = np.fromfile(path / "command_bytes.bin", dtype=np.uint8)
    batch_bytes = np.fromfile(path / "batch_bytes.bin", dtype=np.uint8)

    return validate_payload(
        RequestPayload(
            image=np.asarray(image, dtype=np.uint8),
            command_version=command_version,
            command_bytes=command_bytes,
            batch_bytes=batch_bytes,
            label=path.name,
        )
    )


def validate_payload(payload: RequestPayload) -> RequestPayload:
    image = np.asarray(payload.image, dtype=np.uint8)
    if image.shape == (1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
        image = image[0]
    if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
        raise ValueError(
            f"{payload.label}: input_images must have shape "
            f"({IMAGE_HEIGHT}, {IMAGE_WIDTH}, {IMAGE_CHANNELS}) or "
            f"(1, {IMAGE_HEIGHT}, {IMAGE_WIDTH}, {IMAGE_CHANNELS}), got {image.shape}"
        )

    command_version = np.asarray(payload.command_version, dtype=np.int32).reshape(-1)
    if command_version.shape == (0,):
        raise ValueError(f"{payload.label}: command_version must contain one value")
    if command_version.shape != (1,):
        raise ValueError(
            f"{payload.label}: command_version must have shape (1,), got {command_version.shape}"
        )

    command_bytes = np.asarray(payload.command_bytes, dtype=np.uint8).reshape(-1)
    batch_bytes = np.asarray(payload.batch_bytes, dtype=np.uint8).reshape(-1)

    return RequestPayload(
        image=image,
        command_version=command_version,
        command_bytes=command_bytes,
        batch_bytes=batch_bytes,
        label=payload.label,
    )


def build_synthetic_image(index: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "zeros":
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)

    if mode == "noise":
        return rng.integers(
            0,
            256,
            size=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
            dtype=np.uint8,
        )

    x = np.linspace(0.0, 1.0, IMAGE_WIDTH, dtype=np.float32)
    y = np.linspace(0.0, 1.0, IMAGE_HEIGHT, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    r = np.mod(xx * 255.0 + index * 17.0, 256.0)
    g = np.mod(yy * 255.0 + index * 31.0, 256.0)
    b = np.mod((xx + yy) * 127.0 + index * 47.0, 256.0)
    image = np.stack([r, g, b], axis=-1)
    return image.astype(np.uint8, copy=False)


def build_synthetic_payloads(
    count: int,
    mode: str,
    seed: int,
) -> list[RequestPayload]:
    rng = np.random.default_rng(seed)
    payloads: list[RequestPayload] = []
    for index in range(count):
        payloads.append(
            RequestPayload(
                image=build_synthetic_image(index, mode, rng),
                command_version=np.asarray([COMMAND_ABI_VERSION], dtype=np.int32),
                command_bytes=np.zeros((0,), dtype=np.uint8),
                batch_bytes=np.zeros((0,), dtype=np.uint8),
                label=f"synthetic_{index}",
            )
        )
    return payloads


def load_payloads(args: argparse.Namespace) -> list[RequestPayload]:
    payloads: list[RequestPayload] = []
    for npz_path in args.request_npz:
        payloads.append(load_payload_from_npz(Path(npz_path)))
    for dir_path in args.request_dir:
        payloads.append(load_payload_from_dir(Path(dir_path)))

    if payloads:
        return payloads

    return build_synthetic_payloads(
        count=max(1, args.synthetic_requests),
        mode=args.synthetic_mode,
        seed=args.seed,
    )


def percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


async def run_phase(
    client: TritonGpuBenchmarkClient,
    payloads: Sequence[RequestPayload],
    concurrency: int,
    duration_s: float,
    max_requests: int,
    collect_latency: bool,
) -> BenchmarkStats:
    stop_at = time.perf_counter() + duration_s
    payload_count = len(payloads)
    payload_index = 0
    stats = BenchmarkStats()
    lock = asyncio.Lock()

    async def claim_payload() -> tuple[int, RequestPayload] | None:
        nonlocal payload_index
        async with lock:
            now = time.perf_counter()
            if now >= stop_at:
                return None
            if max_requests > 0 and stats.request_count >= max_requests:
                return None
            idx = payload_index
            payload_index += 1
            stats.request_count += 1
            return idx, payloads[idx % payload_count]

    async def commit_success(latency_s: float, output_bytes: int) -> None:
        async with lock:
            stats.success_count += 1
            stats.total_latency_s += latency_s
            stats.output_bytes += output_bytes
            if collect_latency:
                stats.latencies_s.append(latency_s)

    async def commit_failure(message: str) -> None:
        async with lock:
            stats.failure_count += 1
            if not stats.first_error:
                stats.first_error = message

    async def worker() -> None:
        while True:
            claimed = await claim_payload()
            if claimed is None:
                return
            _, payload = claimed
            started = time.perf_counter()
            try:
                output = await client.infer(payload)
            except Exception as exc:
                await commit_failure(str(exc))
                continue

            latency_s = time.perf_counter() - started
            await commit_success(latency_s, int(output.nbytes))

    tasks = [asyncio.create_task(worker()) for _ in range(max(1, concurrency))]
    await asyncio.gather(*tasks)
    return stats


def print_payload_summary(payloads: Sequence[RequestPayload]) -> None:
    print("Loaded request payloads")
    print(f"  count               : {len(payloads)}")
    for payload in payloads[:5]:
        print(
            "  sample              : "
            f"{payload.label}, image={payload.image.shape}, "
            f"command_bytes={payload.command_bytes.size}, "
            f"batch_bytes={payload.batch_bytes.size}"
        )
    if len(payloads) > 5:
        print(f"  sample              : ... {len(payloads) - 5} more")


def print_results(
    measured_s: float,
    stats: BenchmarkStats,
    payload_count: int,
    concurrency: int,
) -> None:
    success_rate = (
        100.0 * float(stats.success_count) / float(stats.request_count)
        if stats.request_count
        else 0.0
    )
    throughput_rps = (
        float(stats.success_count) / measured_s if measured_s > 0.0 else 0.0
    )
    mpix_per_s = throughput_rps * (IMAGE_HEIGHT * IMAGE_WIDTH) / 1_000_000.0
    mean_latency_ms = (
        (stats.total_latency_s / stats.success_count) * 1000.0
        if stats.success_count
        else 0.0
    )
    sorted_latencies = sorted(stats.latencies_s)
    p50_ms = percentile(sorted_latencies, 0.50) * 1000.0
    p90_ms = percentile(sorted_latencies, 0.90) * 1000.0
    p95_ms = percentile(sorted_latencies, 0.95) * 1000.0
    p99_ms = percentile(sorted_latencies, 0.99) * 1000.0
    stdev_ms = (
        statistics.pstdev(sorted_latencies) * 1000.0
        if len(sorted_latencies) > 1
        else 0.0
    )

    print("\nBenchmark results")
    print(f"  concurrency         : {concurrency}")
    print(f"  payload variants    : {payload_count}")
    print(f"  measured seconds    : {measured_s:.3f}")
    print(f"  requests attempted  : {stats.request_count}")
    print(f"  requests succeeded  : {stats.success_count}")
    print(f"  requests failed     : {stats.failure_count}")
    print(f"  success rate        : {success_rate:.2f}%")
    print(f"  throughput req/s    : {throughput_rps:.2f}")
    print(f"  throughput img/s    : {throughput_rps:.2f}")
    print(f"  throughput MPix/s   : {mpix_per_s:.2f}")
    print(f"  mean latency ms     : {mean_latency_ms:.2f}")
    print(f"  p50 latency ms      : {p50_ms:.2f}")
    print(f"  p90 latency ms      : {p90_ms:.2f}")
    print(f"  p95 latency ms      : {p95_ms:.2f}")
    print(f"  p99 latency ms      : {p99_ms:.2f}")
    print(f"  latency stdev ms    : {stdev_ms:.2f}")
    if stats.first_error:
        print(f"  first error         : {stats.first_error}")


async def async_main(args: argparse.Namespace) -> int:
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if args.warmup < 0.0:
        raise ValueError("--warmup must be non-negative")
    if args.synthetic_requests <= 0:
        raise ValueError("--synthetic-requests must be positive")

    payloads = load_payloads(args)
    print_payload_summary(payloads)

    if args.save_sample_npz:
        sample_path = Path(args.save_sample_npz)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample = payloads[0]
        np.savez(
            sample_path,
            input_images=sample.image,
            command_version=sample.command_version,
            command_bytes=sample.command_bytes,
            batch_bytes=sample.batch_bytes,
        )
        print(f"Saved sample payload template to {sample_path}")

    client = TritonGpuBenchmarkClient(
        url=args.url,
        model_name=args.model_name,
        timeout_s=args.timeout,
    )
    if not await client.is_ready():
        raise RuntimeError(f"Triton server is not reachable at {args.url}")

    try:
        if args.warmup > 0.0:
            print(f"\nWarmup for {args.warmup:.2f}s with concurrency={args.concurrency}")
            await run_phase(
                client=client,
                payloads=payloads,
                concurrency=args.concurrency,
                duration_s=args.warmup,
                max_requests=0,
                collect_latency=False,
            )

        print(f"\nMeasured run for {args.duration:.2f}s with concurrency={args.concurrency}")
        started = time.perf_counter()
        stats = await run_phase(
            client=client,
            payloads=payloads,
            concurrency=args.concurrency,
            duration_s=args.duration,
            max_requests=args.max_requests,
            collect_latency=True,
        )
        measured_s = time.perf_counter() - started
        print_results(
            measured_s=measured_s,
            stats=stats,
            payload_count=len(payloads),
            concurrency=args.concurrency,
        )
        return 0 if stats.failure_count == 0 else 2
    finally:
        await client.close()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
