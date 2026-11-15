"""User-facing error messages (Bahasa Indonesia) that never leak internals."""
from __future__ import annotations

import logging

logger = logging.getLogger("verigraph")

_NETWORK = ("ConnectionError", "Timeout", "ConnectTimeout", "ReadTimeout", "HTTPError", "SSLError")


def friendly_error(exc: BaseException, action: str) -> str:
    """Return a short, safe message for ``action``; log the full detail.

    Validation errors raised on purpose by our own code (``ValueError``) carry
    messages written for users, so they are shown as is. Everything else is
    replaced by a generic hint to avoid exposing paths or stack details.
    """
    logger.warning("%s failed", action, exc_info=exc)
    name = type(exc).__name__
    if isinstance(exc, ValueError):
        return f"{action} gagal: {exc}"
    if isinstance(exc, FileNotFoundError):
        return f"{action} gagal: berkas yang dibutuhkan tidak ditemukan."
    if isinstance(exc, PermissionError):
        return f"{action} gagal: tidak ada izin menulis/membaca berkas."
    if name in _NETWORK or isinstance(exc, (ConnectionError, TimeoutError)):
        return f"{action} gagal: koneksi jaringan bermasalah. Coba lagi beberapa saat."
    if isinstance(exc, (ImportError, RuntimeError)) and "requirements" in str(exc):
        return f"{action} gagal: paket opsional belum terpasang ({exc})."
    if isinstance(exc, ImportError):
        return f"{action} gagal: paket opsional belum terpasang. Lihat README bagian instalasi."
    return f"{action} gagal karena kesalahan tak terduga. Periksa input lalu coba lagi."
