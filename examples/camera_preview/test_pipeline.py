import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageCms, PngImagePlugin

from examples.camera_preview.pipeline import (
    DECODE_SRGB, PixelEngine, Preset, encode_srgb, find_frames, load_frame, preview_batch,
)


class CameraPreviewTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def frame(self, filename="frame.png", size=(5, 3), mode="RGB", color=None, **kwargs):
        path = self.root / filename
        Image.new(mode, size, color).save(path, **kwargs)
        return path

    def test_srgb_roundtrip(self):
        samples = np.arange(256, dtype=np.uint8)
        np.testing.assert_array_equal(encode_srgb(DECODE_SRGB[samples]), samples)
        np.testing.assert_array_equal(encode_srgb(np.array([-1, 0, 1, 2], dtype=np.float32)),
                                      [0, 0, 255, 255])
        self.assertAlmostEqual(DECODE_SRGB[128], .2158605)

    def test_presets_reject_nonfinite_and_out_of_range(self):
        for stops, lift in [(float("nan"), 0), (float("inf"), 0), (-9, 0), (9, 0),
                            (0, float("nan")), (0, -.01), (0, 1.01)]:
            with self.subTest(stops=stops, lift=lift), self.assertRaises(ValueError):
                Preset(stops, lift)

    def test_orientation_and_icc_profile(self):
        exif = Image.Exif()
        exif[274] = 6
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        path = self.frame("rotate.jpg", size=(7, 2), color=(40, 50, 60), exif=exif, icc_profile=profile)
        self.assertEqual(load_frame(path).shape, (7, 2, 3))
        invalid = self.frame("bad-profile.png", icc_profile=b"invalid ICC profile")
        with self.assertRaisesRegex(ValueError, "Embedded profile"):
            load_frame(invalid)

    def test_non_srgb_metadata_fails_explicitly(self):
        for chunk, value in [(b"gAMA", (100_000).to_bytes(4, "big")), (b"cICP", bytes((9, 16, 0, 1)))]:
            with self.subTest(chunk=chunk):
                info = PngImagePlugin.PngInfo()
                info.add(chunk, value)
                path = self.frame(pnginfo=info)
                with self.assertRaisesRegex(ValueError, "sRGB"):
                    load_frame(path)

    def test_invalid_depth_transparency_and_animation(self):
        sixteen_bit = self.root / "sixteen.png"
        Image.fromarray(np.array([[65535, 22]], dtype=np.uint16)).save(sixteen_bit)
        alpha = self.frame("alpha.png", mode="RGBA")
        transparent = self.frame("hidden.png", transparency=(1, 2, 3))
        animated = self.frame("animated.png", color=(0, 0, 0), save_all=True,
                              append_images=[Image.new("RGB", (5, 3), (20, 40, 60))])
        for path, error in ((sixteen_bit, "8-bit PNG"), (alpha, "transparency"),
                            (transparent, "transparency"), (animated, "single still")):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, error):
                load_frame(path)

    def test_pillow_size_limit_is_an_error(self):
        path = self.frame(size=(4, 3))
        with patch.object(Image, "MAX_IMAGE_PIXELS", 10), self.assertRaisesRegex(ValueError, "pixels"):
            load_frame(path)

    def test_directory_order_and_input_types(self):
        frame_b = self.frame("z.JPEG")
        frame_a = self.frame("a.png")
        (self.root / "sidecar.txt").write_text("notes")
        inner = self.root / "other"
        inner.mkdir()
        with self.assertRaisesRegex(ValueError, "expected an image"):
            find_frames(inner)
        self.assertEqual(find_frames(self.root), [frame_a, frame_b])
        self.assertEqual(find_frames(frame_b), [frame_b])

    def test_grayscale_clips_and_output_has_srgb_tag(self):
        path = self.frame(mode="L", color=255)
        before = path.read_bytes()
        out = self.root / "preview"
        report = preview_batch([path], out, Preset(1, 0), "numpy")
        self.assertEqual(report["frames"][0]["highlight_clips"], 15)
        self.assertEqual(before, path.read_bytes())
        self.assertEqual(json.loads((out / "report.json").read_text()), report)
        with Image.open(out / "frame.png.preview.png") as image:
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.info["srgb"], 0)
            np.testing.assert_array_equal(np.asarray(image), np.full((3, 5), 255))

    def test_failed_batch_leaves_no_partial_output(self):
        first = self.frame("a.png")
        broken = self.root / "broken.png"
        broken.write_bytes(b"\x89PNG\r\n\x1a\ncut")
        out = self.root / "preview"
        with self.assertRaisesRegex(ValueError, "broken.png"):
            preview_batch([first, broken], out, Preset(), "numpy")
        self.assertFalse(out.exists())
        self.assertEqual(list(self.root.glob(".forge-preview-*")), [])

    def test_existing_output_and_case_collisions_fail_before_writing(self):
        path = self.frame("a.png")
        out = self.root / "preview"
        out.mkdir()
        existing = out / "existing.txt"
        existing.write_text("keep me")
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            preview_batch([path], out, Preset(), "numpy")
        self.assertEqual(existing.read_text(), "keep me")
        with self.assertRaisesRegex(ValueError, "collide"):
            preview_batch([path, self.root / "A.PNG"], out, Preset(), "numpy")

    def test_odd_sizes_and_shape_reuse_match_reference(self):
        arrays = [DECODE_SRGB[np.arange(count, dtype=np.uint8)] for count in (7, 45, 7, 45)]
        reference = PixelEngine("numpy", Preset())
        for backend, loops in [("forge", 1), ("unfused", 4)]:
            engine = PixelEngine(backend, Preset())
            for arr in arrays:
                np.testing.assert_array_equal(engine.prepare(arr)().view(np.uint32),
                                              reference.prepare(arr)().view(np.uint32))
            self.assertEqual(engine.compilations, 2)
            for size in (7, 45):
                self.assertIn(f"loops={loops} temporary_arrays={loops - 1}", engine.plans[size])

    def test_bundled_frames_and_all_native_backends(self):
        data = Path(__file__).with_name("data")
        manifest = json.loads((data / "manifest.json").read_text())
        paths = find_frames(data / "frames")
        self.assertEqual([(path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
                         [(record["file"], record["sha256"]) for record in manifest["frames"]])
        expected = None
        for backend in ("numpy", "forge", "unfused"):
            out = self.root / backend
            batch = preview_batch(paths[:2], out, Preset(), backend)
            encoded = [(out / record["output"]).read_bytes() for record in batch["frames"]]
            if expected is None:
                expected = encoded
            self.assertEqual(encoded, expected)
            self.assertEqual(batch["compilations"], int(backend != "numpy"))
