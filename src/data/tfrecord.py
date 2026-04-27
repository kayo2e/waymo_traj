"""Raw TFRecord parser — no TensorFlow required."""

import struct


def iter_tfrecord(path):
    """Yield raw record bytes from a TFRecord file without importing TensorFlow."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(12)          # uint64 length + uint32 crc
            if len(hdr) < 12:
                break
            length = struct.unpack("<Q", hdr[:8])[0]
            data   = f.read(length)
            f.read(4)                 # uint32 crc
            if len(data) == length:
                yield data


def iter_tfrecords(paths):
    """Yield raw record bytes from a list of TFRecord files in order."""
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        yield from iter_tfrecord(p)
