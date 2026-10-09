# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from io import BytesIO

import msgspec
import numpy as np
import pytest
from PIL import Image

from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackDecoder, OmniMsgpackEncoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def roundtrip(image):
    prompt = {"multi_modal_data": {"image": [image]}}
    restored = OmniMsgpackDecoder().decode(OmniMsgpackEncoder().encode(prompt))
    return restored["multi_modal_data"]["image"][0]


@pytest.mark.parametrize("transparency", [0, bytes([0, 128, 255])])
def test_palette_png_keeps_colors_and_transparency(transparency):
    image = Image.new("P", (3, 1))
    image.putdata([0, 1, 2])
    image.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255])
    png = BytesIO()
    image.save(png, format="PNG", transparency=transparency)
    png.seek(0)
    image = Image.open(png)

    restored = roundtrip(image)

    assert restored.mode == "P"
    assert restored.tobytes() == image.tobytes()
    assert restored.convert("RGBA").tobytes() == image.convert("RGBA").tobytes()


@pytest.mark.parametrize("mode", ["1", "L", "RGB", "RGBA", "I;16", "I;16B", "I", "F"])
def test_native_pixel_bytes_roundtrip(mode):
    # Non-byte-aligned width exercises PIL's packed mode-1 representation.
    if mode.startswith("I;16"):
        dtype = ">u2" if mode == "I;16B" else "<u2"
        values = np.array(([0, 1024, 32768, 65535] * 5)[:18], dtype=dtype)
        image = Image.frombytes(mode, (9, 2), values.tobytes())
    else:
        image = Image.new(mode, (9, 2))
        image.putdata(([0, 1, 2, 3] * 5)[:18])

    restored = roundtrip(image)

    assert restored.mode == image.mode
    assert restored.size == image.size
    assert restored.tobytes() == image.tobytes()


def test_rgb_packet_remains_readable_by_legacy_decoder():
    image = Image.new("RGB", (2, 1), (12, 34, 56))
    packet = msgspec.msgpack.decode(OmniMsgpackEncoder().encode(image))

    arr = np.frombuffer(packet["data"], dtype=np.uint8).reshape(packet["shape"])
    restored = Image.fromarray(arr, mode=packet["mode"])

    assert restored.tobytes() == image.tobytes()


def test_rgb_transparency_roundtrip():
    image = Image.new("RGB", (2, 1), (12, 34, 56))
    image.info["transparency"] = (12, 34, 56)

    assert roundtrip(image).convert("RGBA").tobytes() == image.convert("RGBA").tobytes()


def test_legacy_rgb_packet_still_decodes():
    image = Image.new("RGB", (2, 1), (12, 34, 56))
    arr = np.asarray(image, dtype=np.uint8)
    packet = {"__pil_image__": True, "mode": image.mode, "shape": list(arr.shape), "data": arr.tobytes()}

    restored = OmniMsgpackDecoder().decode(msgspec.msgpack.encode(packet))

    assert restored.mode == image.mode
    assert restored.tobytes() == image.tobytes()
