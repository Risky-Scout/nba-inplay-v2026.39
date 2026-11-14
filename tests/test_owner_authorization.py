"""Use ephemeral RSA keys and fake HTTP sessions; never load owner credentials."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryptography.hazmat.primitives.asymmetric import rsa, ec
import owner_authorization as OA
import kalshi_nba_inplay_trader as T


class OwnerAuthorizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owner = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.test_pin = OA.public_key_fingerprint(cls.owner)

    def client(self, key, paper=False):
        client = T.KalshiClient("test-key-id", "", paper=paper)
        client._private_key = key
        client._session = Mock()
        client._session.post.return_value = SimpleNamespace(status_code=201, json=lambda: {"order_id": "test-order", "fill_count": 0})
        client._session.delete.return_value = SimpleNamespace(status_code=200)
        return client

    def test_owner_pin_is_supplied_public_fingerprint(self):
        self.assertEqual(OA.OWNER_PUBLIC_KEY_SHA256,
                         "57529816ec253a53d61987d688eb3d534d5a83eb7b04d6b5562ec45dbf3014b3")
        self.assertFalse(OA.is_authorized_signer(self.owner))
        self.assertFalse(OA.is_authorized_signer(self.other))

    def test_matching_key_allows_only_explicit_live_mode(self):
        with patch.object(OA, "OWNER_PUBLIC_KEY_SHA256", self.test_pin):
            self.assertTrue(self.client(self.owner).can_trade)
            self.assertFalse(self.client(self.owner, paper=True).can_trade)
            self.assertFalse(self.client(self.other).can_trade)
            missing_id = self.client(self.owner)
            missing_id.key_id = ""
            self.assertFalse(missing_id.can_trade)

    def test_public_missing_and_wrong_type_keys_cannot_authorize(self):
        with patch.object(OA, "OWNER_PUBLIC_KEY_SHA256", self.test_pin):
            for key in (None, object(), self.owner.public_key(), ec.generate_private_key(ec.SECP256R1())):
                self.assertFalse(OA.is_authorized_signer(key))

    def test_nonowner_cannot_submit_exit_cancel_or_sign_mutation(self):
        with patch.object(OA, "OWNER_PUBLIC_KEY_SHA256", self.test_pin):
            client = self.client(self.other)
            self.assertEqual(client.send_order(None, "yes", 40, 1, True, 8), (None, 0.0))
            self.assertEqual(client.send_exit(None, "yes", 40, 1), (None, 0.0))
            self.assertFalse(client.cancel_order("test-order", "test-market"))
            for method in ("POST", "PUT", "PATCH", "DELETE"):
                with self.assertRaises(PermissionError):
                    client._auth(method, "/portfolio/events/orders")
            client._session.post.assert_not_called()
            client._session.delete.assert_not_called()

    def test_matching_key_reaches_mocked_exchange_and_key_swap_revokes(self):
        with patch.object(OA, "OWNER_PUBLIC_KEY_SHA256", self.test_pin):
            client = self.client(self.owner)
            market = SimpleNamespace(ticker="TEST", exchange_index=None)
            self.assertEqual(client.send_order(market, "yes", 40, 1, True, 8), ("test-order", 0.0))
            self.assertEqual(client.send_exit(market, "yes", 40, 1), ("test-order", 0.0))
            self.assertTrue(client.cancel_order("test-order", "TEST"))
            self.assertEqual(client._session.post.call_count, 2)
            self.assertEqual(client._session.delete.call_count, 1)
            client._private_key = self.other
            self.assertFalse(client.can_trade)
            self.assertEqual(client.send_order(market, "yes", 40, 1, True, 8), (None, 0.0))
            self.assertEqual(client._session.post.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
