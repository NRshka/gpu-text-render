#!/usr/bin/env python3
import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from gpu_font_planner import GpuRenderPlanner
except ImportError as exc:  # pragma: no cover - import failure is surfaced as skip reason
    GpuRenderPlanner = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""


ATLAS_MAGIC = 0x46415431
ATLAS_VERSION_2 = 2
IMAGE_HEIGHT = 1200
IMAGE_WIDTH = 900


def write_test_atlas(path: Path) -> None:
    header = struct.pack(
        "<8I",
        ATLAS_MAGIC,
        ATLAS_VERSION_2,
        8,
        8,
        2,
        32 + 2 * 24,
        16,
        8,
    )

    glyph_a = struct.pack(
        "<IHHHHfff",
        ord("A"),
        0,
        0,
        2,
        2,
        3.0,
        0.0,
        2.0,
    )
    glyph_space = struct.pack(
        "<IHHHHfff",
        ord(" "),
        0,
        0,
        0,
        0,
        1.0,
        0.0,
        0.0,
    )

    pixels = bytearray(8 * 8)
    pixels[0] = 255
    pixels[1] = 255
    pixels[8] = 255
    pixels[9] = 255

    path.write_bytes(header + glyph_a + glyph_space + bytes(pixels))


def make_test_image() -> np.ndarray:
    return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)


@unittest.skipUnless(GpuRenderPlanner is not None, f"gpu_font_planner not importable: {IMPORT_ERROR}")
class GpuRenderPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        atlas_dir = Path(self._tmpdir.name)
        write_test_atlas(atlas_dir / "atlas_16.bin")
        self.planner = GpuRenderPlanner(str(atlas_dir))
        self.image = make_test_image()
        self.texts = ["A"]
        self.polygons = [[[0.0, 0.0], [100.0, 0.0], [100.0, 40.0], [0.0, 40.0]]]

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_smoke_plan_returns_nonempty_byte_buffers(self) -> None:
        command_bytes, batch_bytes = self.planner.plan(
            self.image,
            self.texts,
            self.polygons,
        )

        self.assertIsInstance(command_bytes, bytes)
        self.assertIsInstance(batch_bytes, bytes)
        self.assertGreater(len(command_bytes), 0)
        self.assertGreater(len(batch_bytes), 0)

    def test_plan_accepts_explicit_profile(self) -> None:
        command_bytes, batch_bytes = self.planner.plan(
            self.image,
            self.texts,
            self.polygons,
            profile="quality",
        )

        self.assertIsInstance(command_bytes, bytes)
        self.assertIsInstance(batch_bytes, bytes)
        self.assertGreater(len(command_bytes), 0)
        self.assertGreater(len(batch_bytes), 0)

    def test_plan_accepts_cluster_ids(self) -> None:
        command_bytes, batch_bytes = self.planner.plan(
            self.image,
            self.texts,
            self.polygons,
            cluster_ids=[0],
        )

        self.assertIsInstance(command_bytes, bytes)
        self.assertIsInstance(batch_bytes, bytes)
        self.assertGreater(len(command_bytes), 0)
        self.assertGreater(len(batch_bytes), 0)

    def test_validation_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            self.planner.plan(np.zeros((100, 100, 3), dtype=np.uint8), self.texts, self.polygons)

        with self.assertRaises(ValueError):
            self.planner.plan(self.image, ["A", "B"], self.polygons)

        with self.assertRaises(ValueError):
            self.planner.plan(self.image, self.texts, [[[0.0, 0.0], [1.0, 0.0]]])

        with self.assertRaises(ValueError):
            self.planner.plan(
                self.image,
                self.texts,
                self.polygons,
                original_texts=["x", "y"],
            )

        with self.assertRaises(ValueError):
            self.planner.plan(
                self.image,
                self.texts,
                self.polygons,
                rgba=[0xFFFFFFFF, 0x11223344],
            )

        with self.assertRaises(ValueError):
            self.planner.plan(
                self.image,
                self.texts,
                self.polygons,
                cluster_ids=[0, 1],
            )

        with self.assertRaises(ValueError):
            self.planner.plan(
                self.image,
                self.texts,
                self.polygons,
                cluster_ids=[-2],
            )

        with self.assertRaises(ValueError):
            self.planner.plan(
                self.image,
                self.texts,
                self.polygons,
                profile="invalid",
            )


@unittest.skipUnless(GpuRenderPlanner is not None, f"gpu_font_planner not importable: {IMPORT_ERROR}")
class GpuRenderPlannerTritonIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._triton_url = os.environ.get("GPU_FONT_TRITON_TEST_URL", "")
        self._atlas_dir = os.environ.get("GPU_FONT_TRITON_TEST_ATLAS_DIR", "")
        if not self._triton_url or not self._atlas_dir:
            self.skipTest(
                "Set GPU_FONT_TRITON_TEST_URL and GPU_FONT_TRITON_TEST_ATLAS_DIR to run integration tests"
            )

        from reference.triton_text_renderer import TritonTextRendererGpuClient

        self.planner = GpuRenderPlanner(self._atlas_dir)
        self.client = TritonTextRendererGpuClient(url=self._triton_url)
        if not await self.client.is_ready():
            self.skipTest(f"Triton server is not reachable at {self._triton_url}")

    async def asyncTearDown(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            await client.close()

    async def test_plan_and_render_shape(self) -> None:
        image = make_test_image()
        command_bytes, batch_bytes = self.planner.plan(
            image,
            ["A"],
            [[[0.0, 0.0], [120.0, 0.0], [120.0, 48.0], [0.0, 48.0]]],
        )
        rendered = await self.client.infer_text_renderer_gpu(
            input_image=image,
            command_bytes=command_bytes,
            batch_bytes=batch_bytes,
        )

        self.assertEqual(rendered.shape, (IMAGE_HEIGHT, IMAGE_WIDTH, 3))


if __name__ == "__main__":
    unittest.main()
