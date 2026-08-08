####################################################################################################
#                                          download.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 2026-08-08                                                                              #
#                                                                                                  #
# Purpose: Fetching a large file once and keeping it. Public datasets are big, served by hosts     #
#          that are not always well behaved, and worth downloading exactly one time - so this      #
#          verifies what it got, survives a stalled connection, and caches the result.             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import hashlib
import os
import re
import ssl
import urllib.parse
import urllib.request
from pathlib import Path


__all__ = ['cache_root', 'fetch']


def cache_root() -> Path:
    """
    Directory downloaded data is kept in.

    Set "AUGMENTRUM_CACHE" to move it; otherwise the usual user cache location.
    """
    root = os.environ.get('AUGMENTRUM_CACHE')
    return Path(root) if root else Path.home() / '.cache' / 'augmentrum'


#**************#
#   fetching   #
#**************#
def fetch(url: str, target: Path, md5: str = None, size: int = None,
          progress: bool = True, timeout: int = 120, attempts: int = 3) -> Path:
    """
    Download *url* to *target*, verifying what arrives.

    The file is written under a temporary name and moved into place only once
    it is known to be complete, so an interrupted download is never mistaken
    for a finished one.

    Args:
        url: What to fetch.
        target: Where it ends up.
        md5: Expected checksum. The download is discarded if it does not match.
        size: Expected length in bytes, used to notice a connection that ends
            early and to report progress against.
        progress: Report progress to stdout.
        timeout: Seconds a read may stall before the attempt is abandoned.
            Without one a dead connection blocks forever.
        attempts: How many times to try. A host serving no byte ranges cannot
            resume, so an interrupted download starts over - worth repeating
            once or twice before giving up on several gigabytes.

    Returns:
        *target*.

    Raises:
        OSError: If every attempt ended before the file did.
        RuntimeError: If the checksum does not match. Nothing is kept.
    """
    partial = target.with_suffix(target.suffix + '.part')
    context = _ssl_context(urllib.parse.urlsplit(url).hostname)

    for attempt in range(1, attempts + 1):
        try:
            digest = _stream(url, context, partial, size, progress, timeout)
            break
        except OSError as error:
            partial.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            if progress:
                print(f"  attempt {attempt} stopped ({error}); starting over", flush=True)

    if md5 and digest.hexdigest() != md5:
        partial.unlink()
        raise RuntimeError(
            f"{target.name} does not match its published checksum "
            f"({digest.hexdigest()} vs {md5}). Nothing was kept."
        )

    partial.rename(target)
    return target


def _stream(url, context, partial, size, progress, timeout):
    """One attempt at the whole file, returning its running digest."""
    digest = hashlib.md5()

    with urllib.request.urlopen(url, context=context, timeout=timeout) as response, \
            open(partial, 'wb') as out:
        seen = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            digest.update(block)
            seen += len(block)
            if progress and seen % (1 << 28) < (1 << 20):
                total = f" / {size / 1e9:.1f}" if size else ""
                print(f"  {seen / 1e9:5.1f}{total} GB", flush=True)

    if size and seen != size:
        raise OSError(f"connection ended at {seen / 1e9:.1f} of {size / 1e9:.1f} GB")
    return digest


#******************#
#   certificates   #
#******************#
def _ssl_context(host):
    """
    Verification context for talking to *host*.

    Two things get in the way of a plain default context. A Python install does
    not necessarily reach the system certificate store, so certifi's bundle is
    preferred when it is there. And a server is meant to send every certificate
    between its own and a trusted root, but some send only their own and leave
    the client to follow the pointer inside it - which browsers and curl do,
    and Python does not.

    Following that pointer keeps verification real rather than turning it off:
    whatever comes back still has to be signed by a root that was already
    trusted, so a wrong or hostile answer fails the handshake instead of
    passing it.
    """
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()

    missing = _missing_intermediates(host)
    if missing:
        try:
            context.load_verify_locations(cadata=missing)
        except ssl.SSLError:
            # Whatever came back was not a certificate. The chain may well be
            # complete anyway, so verification proceeds on the roots alone
            # rather than the download failing here.
            pass
    return context


def _missing_intermediates(host):
    """
    Certificates *host* leaves out of its chain, or "" when the chain is whole.

    The addresses are read straight out of the leaf certificate, where they sit
    as plain text in its Authority Information Access extension.
    """
    try:
        leaf = ssl.get_server_certificate((host, 443))
    except (OSError, ssl.SSLError):
        return ''

    chain = ''
    for url in re.findall(rb'http://[a-zA-Z0-9./_-]+\.(?:cer|crt|der|p7c)',
                          ssl.PEM_cert_to_DER_cert(leaf)):
        try:
            with urllib.request.urlopen(url.decode(), timeout=30) as response:
                chain += ssl.DER_cert_to_PEM_cert(response.read())
        except (OSError, ValueError):
            continue
    return chain
