####################################################################################################
#                                       test_download.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Holds the cached download to the one thing that matters when a file is several          #
#          gigabytes: what it keeps is what was published, and what is not, it does not keep.      #
#                                                                                                  #
####################################################################################################

"""
Tests for fetching a large file once.

None of these reach the network. What is worth guarding is the failure paths -
a stalled connection, a truncated file, a file that arrives whole but wrong -
because those are the ones that would otherwise leave a broken cache behind and
be believed forever after.
"""

#*************#
#   imports   #
#*************#
import io
import urllib.request

import pytest

from augmentrum.utils import download


#*************#
#   helpers   #
#*************#
def _serving(payload, monkeypatch):
    """Point the downloader at *payload* instead of the network."""
    monkeypatch.setattr(download, '_ssl_context', lambda host: None)
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(payload))


#**************#
#   fetching   #
#**************#
def test_a_complete_download_is_kept(tmp_path, monkeypatch):
    """The straightforward case, so the failure cases mean something."""
    import hashlib

    payload = b'a whole archive, honestly'
    _serving(payload, monkeypatch)

    target = tmp_path / 'archive.zip'
    download.fetch('https://example.invalid/archive.zip', target,
                   md5=hashlib.md5(payload).hexdigest(), size=len(payload),
                   progress=False)

    assert target.read_bytes() == payload
    assert not list(tmp_path.glob('*.part')), "the temporary name was left behind"


def test_a_truncated_download_is_not_kept(tmp_path, monkeypatch):
    """Half a file that looks whole would be worse than no file."""
    _serving(b'only the first few bytes', monkeypatch)

    target = tmp_path / 'archive.zip'
    with pytest.raises(OSError, match="connection ended at"):
        download.fetch('https://example.invalid/archive.zip', target,
                       size=5_000_000_000, progress=False, attempts=1)

    assert not target.exists()
    assert not list(tmp_path.glob('*.part')), "a failed download left a partial file"


def test_a_truncated_download_is_retried(tmp_path, monkeypatch):
    """A stall cannot be resumed, so the remedy is to start again."""
    attempts = []

    def counting(*args, **kwargs):
        attempts.append(1)
        return io.BytesIO(b'short')

    monkeypatch.setattr(download, '_ssl_context', lambda host: None)
    monkeypatch.setattr(urllib.request, 'urlopen', counting)

    with pytest.raises(OSError):
        download.fetch('https://example.invalid/archive.zip', tmp_path / 'a.zip',
                       size=999, progress=False, attempts=3)

    assert len(attempts) == 3


def test_a_download_that_fails_its_checksum_is_not_kept(tmp_path, monkeypatch):
    """Right length, wrong bytes: the checksum is what catches that."""
    payload = b'the wrong archive entirely'
    _serving(payload, monkeypatch)

    target = tmp_path / 'archive.zip'
    with pytest.raises(RuntimeError, match="does not match its published checksum"):
        download.fetch('https://example.invalid/archive.zip', target,
                       md5='0' * 32, size=len(payload), progress=False)

    assert not target.exists()
    assert not list(tmp_path.glob('*.part')), "a failed download left a partial file"


#*************#
#   caching   #
#*************#
def test_the_cache_location_can_be_moved(tmp_path, monkeypatch):
    """A shared machine should not be forced to write to a home directory."""
    monkeypatch.setenv('AUGMENTRUM_CACHE', str(tmp_path))
    assert download.cache_root() == tmp_path

    monkeypatch.delenv('AUGMENTRUM_CACHE')
    assert download.cache_root().name == 'augmentrum'
