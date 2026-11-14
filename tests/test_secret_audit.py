"""Credential detection must report locations without reproducing values."""
import importlib.util
import json
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1]/"scripts"/"audit_repository_secrets.py"
spec = importlib.util.spec_from_file_location("secret_audit", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class SecretAuditTests(unittest.TestCase):
    def test_private_key_detection_without_disclosure(self):
        marker = b"-"*5+b"BEGIN "+b"RSA PRIVATE KEY"+b"-"*5
        hits = audit.findings(b"header\n"+marker+b"\nprivate material")
        self.assertEqual(hits, [dict(rule="private-key", line=2)])
        self.assertNotIn("private material", json.dumps(hits))

    def test_token_values_never_appear_in_report(self):
        token = b"ghp"+b"_"+b"A"*36
        hits = audit.findings(token)
        self.assertEqual(hits[0]["rule"], "github-token")
        self.assertNotIn(token.decode(), json.dumps(hits))

    def test_variable_references_are_not_credentials(self):
        text = b'api_key = os.getenv("DT_API_KEY")\nAuthorization: ${SECRET}\n'
        self.assertEqual(audit.findings(text), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
