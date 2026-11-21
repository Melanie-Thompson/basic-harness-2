"""Package signing — Ed25519 signatures for tool packages.

The API owner signs the tool package (definition + code) with their private
key. The harness verifies the signature against the owner's public key before
loading any code into the sandbox.

Signing flow (publisher side):
    1. Serialize the tool definition + source code into a canonical payload
    2. Sign with the owner's Ed25519 private key
    3. Ship the signature alongside the package

Verification flow (harness side):
    1. Load the package but don't execute anything
    2. Reconstruct the canonical payload from the definition + source
    3. Verify the signature against the owner's public key
    4. If valid, load into sandbox. If not, reject the package.
"""

import hashlib
import json
import logging
from base64 import b64decode, b64encode
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

log = logging.getLogger("package_signing")


@dataclass
class PackageSignature:
    """A signed tool package."""
    publisher: str
    public_key: str
    signature: str
    payload_hash: str


def generate_keypair() -> tuple[str, str]:
    """Generate an Ed25519 keypair for a tool publisher.

    Returns (private_key_pem, public_key_pem).
    """
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption(),
    ).decode()
    public_pem = private_key.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return private_pem, public_pem


def _canonical_payload(definition_json: str, source_code: str) -> bytes:
    """Build the canonical payload that gets signed.

    The payload is: SHA-256(definition_json) || SHA-256(source_code),
    ensuring both parts are covered and the payload is fixed-size.
    """
    def_hash = hashlib.sha256(definition_json.encode()).digest()
    src_hash = hashlib.sha256(source_code.encode()).digest()
    return def_hash + src_hash


def _definition_to_json(defn: dict) -> str:
    """Serialize a tool definition to canonical JSON for signing.

    Only the fields that matter for the contract are included:
    name, description, inputSchema, kind, placement, capabilities,
    requires_confirmation, depends_on, tunnel.
    """
    canonical = {
        "name": defn.get("name", ""),
        "description": defn.get("description", ""),
        "inputSchema": defn.get("inputSchema", {}),
        "kind": defn.get("kind", "processor"),
        "placement": defn.get("placement", "local"),
        "capabilities": defn.get("capabilities", []),
        "requires_confirmation": defn.get("requires_confirmation", False),
        "depends_on": defn.get("depends_on", []),
    }
    tunnel = defn.get("tunnel")
    if tunnel:
        canonical["tunnel"] = {
            "base_url": tunnel.get("base_url", ""),
            "allowed_hosts": tunnel.get("allowed_hosts", []),
        }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def sign_package(
    definition: dict,
    source_code: str,
    private_key_pem: str,
    publisher: str,
) -> PackageSignature:
    """Sign a tool package with the publisher's Ed25519 private key.

    Args:
        definition: Tool definition as a dict (from ToolDefinition.model_dump())
        source_code: The tool's source code
        private_key_pem: Publisher's Ed25519 private key in PEM format
        publisher: Publisher identifier (e.g. "weather-api.example.com")

    Returns:
        PackageSignature with the signature and public key
    """
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    private_key = load_pem_private_key(private_key_pem.encode(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError("key must be Ed25519")

    def_json = _definition_to_json(definition)
    payload = _canonical_payload(def_json, source_code)
    signature = private_key.sign(payload)

    public_pem = private_key.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo,
    ).decode()

    return PackageSignature(
        publisher=publisher,
        public_key=public_pem,
        signature=b64encode(signature).decode(),
        payload_hash=hashlib.sha256(payload).hexdigest(),
    )


def verify_package(
    definition: dict,
    source_code: str,
    sig: PackageSignature,
    trusted_keys: dict[str, str] | None = None,
) -> bool:
    """Verify a tool package's signature.

    Args:
        definition: Tool definition as a dict
        source_code: The tool's source code
        sig: The PackageSignature to verify
        trusted_keys: Optional dict of publisher -> public_key_pem.
            If provided, the signature's public key must match the
            trusted key for that publisher. If not provided, the
            key embedded in the signature is used (trust-on-first-use).

    Returns:
        True if the signature is valid, False otherwise.
    """
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    public_pem = sig.public_key
    if trusted_keys:
        trusted = trusted_keys.get(sig.publisher)
        if trusted is None:
            log.warning("publisher '%s' not in trusted keys", sig.publisher)
            return False
        if trusted != public_pem:
            log.warning("public key mismatch for publisher '%s'", sig.publisher)
            return False

    try:
        public_key = load_pem_public_key(public_pem.encode())
    except Exception:
        log.warning("invalid public key for publisher '%s'", sig.publisher)
        return False

    if not isinstance(public_key, Ed25519PublicKey):
        log.warning("key for publisher '%s' is not Ed25519", sig.publisher)
        return False

    def_json = _definition_to_json(definition)
    payload = _canonical_payload(def_json, source_code)

    try:
        public_key.verify(b64decode(sig.signature), payload)
        log.info("package '%s' by '%s' verified", definition.get("name"), sig.publisher)
        return True
    except InvalidSignature:
        log.warning("invalid signature for package '%s' by '%s'",
                    definition.get("name"), sig.publisher)
        return False


def verify_package_file(
    definition: dict,
    source_path: str | Path,
    sig: PackageSignature,
    trusted_keys: dict[str, str] | None = None,
) -> bool:
    """Verify a tool package where the source code is in a file."""
    source_code = Path(source_path).read_text()
    return verify_package(definition, source_code, sig, trusted_keys)
