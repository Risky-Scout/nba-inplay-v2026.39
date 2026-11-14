"""Owner signing-key authorization; this module contains only a PUBLIC fingerprint.

SHA-256 is over DER SubjectPublicKeyInfo, matching:
openssl pkey -in local-private.pem -pubout -outform DER | shasum -a 256

Possession of this hash grants no exchange access. The matching private RSA key
must remain local. This pins the unmodified application, not other people's forks.
"""
import hashlib
import hmac

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

OWNER_PUBLIC_KEY_SHA256 = "57529816ec253a53d61987d688eb3d534d5a83eb7b04d6b5562ec45dbf3014b3"


def public_key_fingerprint(private_key):
    """Return a public fingerprint only for an actual RSA private signing key."""
    if not isinstance(private_key, rsa.RSAPrivateKey):
        return None
    try:
        public_der = private_key.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
        return hashlib.sha256(public_der).hexdigest()
    except Exception:
        return None  # Unreadable/unsupported key: refuse authorization without logging key data.


def is_authorized_signer(private_key):
    fingerprint = public_key_fingerprint(private_key)
    return fingerprint is not None and hmac.compare_digest(fingerprint, OWNER_PUBLIC_KEY_SHA256)
