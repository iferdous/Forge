"""Generate review PNGs from ordinary camera frames with a reusable pixel kernel."""

from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import BinaryIO
import warnings

import numpy as np
from PIL import Image, ImageCms, ImageOps, PngImagePlugin


EXTENSIONS = {".png", ".jpg", ".jpeg"}
BACKENDS = ("forge", "unfused", "numpy", "numexpr")
SRGB_CHROMATICITY = (.3127, .3290, .6400, .3300, .3000, .6000, .1500, .0600)

# Each 8-bit input value has just one linear-light value. Build this lookup once.
_srgb_bytes = np.arange(256, dtype=np.float64) / 255
DECODE_SRGB = np.where(_srgb_bytes <= .04045, _srgb_bytes / 12.92,
                       ((_srgb_bytes + .055) / 1.055) ** 2.4).astype(np.float32)


@dataclass(frozen=True)
class Preset:
    stops: float = .35
    lift: float = .015

    def __post_init__(self):
        if not -8 <= self.stops <= 8:
            raise ValueError("Exposure stops must be finite and between -8 and 8")
        if not 0 <= self.lift <= 1:
            raise ValueError("Shadow lift must be finite and between 0 and 1")

    @property
    def coefficients(self):
        return np.float32(2 ** self.stops), np.float32(self.lift), np.float32(1 / (1 + self.lift))

    @property
    def expression(self):
        gain, lift, scale = map(float, self.coefficients)
        return f"sqrt((x * {gain!r} + {lift!r}) * {scale!r})"


def check_png_header(source: BinaryIO):
    if source.read(8) != b"\x89PNG\r\n\x1a\n":
        source.seek(0)
        return
    while header := source.read(8):
        if len(header) != 8:
            break  # Pillow reports a truncated file during decoding.
        count, name = int.from_bytes(header[:4], "big"), header[4:]
        if name in (b"IDAT", b"IEND"):
            break
        if name == b"IHDR" and count == 13:
            if source.read(13)[8:9] != b"\x08":
                raise ValueError("Expected an 8-bit PNG")
            source.seek(4, 1)
        elif name == b"cICP" and count == 4:
            if source.read(4) != bytes((1, 13, 0, 1)):
                raise ValueError("PNG cICP metadata declares a color space other than full-range sRGB")
            source.seek(4, 1)
        else:
            source.seek(count + 4, 1)
    source.seek(0)


def to_srgb(image):
    icc = image.info.get("icc_profile")
    if icc:
        try:
            return ImageCms.profileToProfile(image, BytesIO(icc), ImageCms.createProfile("sRGB"),
                                             outputMode="RGB")
        except (OSError, ImageCms.PyCMSError) as error:
            raise ValueError("Embedded profile could not be converted to sRGB") from error
    if "srgb" not in image.info:
        gamma = image.info.get("gamma", .45455)
        chromaticity = image.info.get("chromaticity", SRGB_CHROMATICITY)
        if not math.isclose(gamma, .45455, abs_tol=.0001) or not np.allclose(
                chromaticity, SRGB_CHROMATICITY, atol=.0001, rtol=0):
            raise ValueError("Convert the image's declared gamma and primaries to sRGB first")
    return image


def load_frame(path: Path):
    """Decode a single oriented 8-bit grayscale or RGB image into sRGB bytes."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with path.open("rb") as source:
                check_png_header(source)
                with Image.open(source, formats=["PNG", "JPEG"]) as image:
                    if getattr(image, "n_frames", 1) != 1:
                        raise ValueError("Expected a single still frame")
                    if image.mode not in ("RGB", "L") or "transparency" in image.info:
                        raise ValueError("Expected 8-bit grayscale or RGB without transparency")
                    return np.asarray(to_srgb(ImageOps.exif_transpose(image)), dtype=np.uint8)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError(f"{path}: {error}") from error


def encode_srgb(linear):
    bounded = np.clip(linear, 0, 1)
    encoded = np.where(bounded <= .0031308, 12.92 * bounded,
                       1.055 * np.power(bounded, 1 / 2.4) - .055)
    return np.rint(encoded * 255).astype(np.uint8)


def save_png(path: Path, pixels):
    info = PngImagePlugin.PngInfo()
    info.add(b"sRGB", b"\x00")  # Explicit sRGB, perceptual rendering intent.
    Image.fromarray(pixels).save(path, format="PNG", pnginfo=info, compress_level=1)


def find_frames(source: Path):
    if source.is_file() and source.suffix.lower() in EXTENSIONS:
        return [source]
    if source.is_dir():
        frames = sorted(path for path in source.iterdir()
                        if path.suffix.lower() in EXTENSIONS and path.is_file())
        if frames:
            return frames
    raise ValueError(f"{source}: expected an image or a folder containing PNG or JPEG frames")


class PixelEngine:
    def __init__(self, backend, preset, target="auto"):
        self.backend = backend
        self.preset = preset
        self.target = None
        self.compile_ns = 0
        self.compilations = 0
        self.plans = {}
        self._coefficients = preset.coefficients
        if backend in ("forge", "unfused"):
            try:
                import forge
                self.target = ("avx2" if forge.has_avx2() else "scalar") if target == "auto" else target
                start = perf_counter_ns()
                self.program = forge.compile(preset.expression, inputs=("x",))
                self.compile_ns = perf_counter_ns() - start
            except (ImportError, AttributeError) as error:
                raise ValueError("Set PYTHONPATH to an LLVM-enabled Forge Python build") from error
        elif backend == "numexpr":
            try:
                import numexpr
            except ImportError as error:
                raise ValueError("Install NumExpr to use the numexpr backend") from error
            numexpr.set_num_threads(1)
            self.numexpr = numexpr
        self._kernels = OrderedDict()
        self._output = np.empty(0, dtype=np.float32)

    def prepare(self, linear):
        # Forge specializes on shape. Flattening a contiguous H×W×3 image
        # keeps the channels packed and adds no copy.
        pixels = linear.reshape(-1)
        if self._output.size != pixels.size:
            self._output = np.empty_like(pixels)
        output = self._output
        if self.target:
            kernel = self.compiled_kernel(pixels)

            def invoke():
                kernel.run_into(output, pixels)
                return output

            return invoke
        if self.backend == "numexpr":
            gain, lift, scale = self._coefficients
            inputs = {"x": pixels, "gain": gain, "lift": lift, "scale": scale}
            return lambda: self.numexpr.evaluate(
                "sqrt((x * gain + lift) * scale)", local_dict=inputs,
                optimization="moderate", out=output)
        gain, lift, scale = self._coefficients

        def numpy_call():
            np.multiply(pixels, gain, out=output)
            np.add(output, lift, out=output)
            np.multiply(output, scale, out=output)
            return np.sqrt(output, out=output)

        return numpy_call

    def compiled_kernel(self, pixels):
        size = pixels.size
        kernel = self._kernels.get(size)
        if kernel is None:
            start = perf_counter_ns()
            kernel = self.program.jit(pixels, target=self.target, fusion=self.backend == "forge")
            self.compile_ns += perf_counter_ns() - start
            self.compilations += 1
            self.plans[size] = kernel.plan.dump()
            self._kernels[size] = kernel
            if len(self._kernels) > 4:
                self._kernels.popitem(last=False)
        self._kernels.move_to_end(size)
        return kernel


def preview_batch(frames, destination, preset, backend="forge", target="auto"):
    output_names = [frame.name + ".preview.png" for frame in frames]
    if len({name.casefold() for name in output_names}) != len(output_names):
        raise ValueError("Input frame names collide in the output folder")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output folder already exists: {destination}")

    total_start = perf_counter_ns()
    engine = PixelEngine(backend, preset, target)
    records = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".forge-preview-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "preview"
        stage.mkdir()
        for frame, name in zip(frames, output_names, strict=True):
            start = perf_counter_ns()
            pixels = load_frame(frame)
            linear = DECODE_SRGB[pixels]
            input_ns = perf_counter_ns() - start
            apply = engine.prepare(linear)
            start = perf_counter_ns()
            adjusted = apply().reshape(pixels.shape)
            kernel_ns = perf_counter_ns() - start
            start = perf_counter_ns()
            clipped = int(np.count_nonzero(adjusted > 1))
            save_png(stage / name, encode_srgb(adjusted))
            export_ns = perf_counter_ns() - start
            records.append({"input": frame.name, "output": name, "shape": list(pixels.shape),
                            "samples": pixels.size, "highlight_clips": clipped,
                            "decode_and_linearize_ns": input_ns, "pixel_ns": kernel_ns,
                            "encode_and_write_ns": export_ns})
        report = {"backend": backend, "target": engine.target, "preset": {
            "stops": preset.stops, "lift": preset.lift}, "expression": preset.expression,
            "compile_ns": engine.compile_ns, "compilations": engine.compilations,
            "plans": engine.plans, "frames": records, "total_ns": perf_counter_ns() - total_start}
        (stage / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Output folder already exists: {destination}")
        stage.rename(destination)
    return report
