import os
import unittest
from unittest.mock import patch
from urllib import request

from duplexconv_stage3.openrouter_client import (
    DEFAULT_NETWORK_ROUTE_POLICY,
    DIRECT_NETWORK_ROUTE_POLICY,
    OpenRouterClient,
)


def proxy_handlers(client):
    return [
        handler
        for handler in client._opener.handlers
        if isinstance(handler, request.ProxyHandler)
    ]


class OpenRouterClientTests(unittest.TestCase):
    def test_environment_policy_respects_selected_proxy_route(self):
        proxy_environment = {
            "HTTP_PROXY": "http://proxy.invalid:1234",
            "HTTPS_PROXY": "http://proxy.invalid:1234",
            "http_proxy": "http://proxy.invalid:1234",
            "https_proxy": "http://proxy.invalid:1234",
        }
        with patch.dict(os.environ, proxy_environment, clear=False):
            client = OpenRouterClient(
                "dummy",
                network_route_policy=DEFAULT_NETWORK_ROUTE_POLICY,
            )
        self.assertTrue(proxy_handlers(client))
        self.assertTrue(proxy_handlers(client)[0].proxies)

    def test_direct_policy_does_not_inherit_proxy_route(self):
        proxy_environment = {
            "HTTP_PROXY": "http://proxy.invalid:1234",
            "HTTPS_PROXY": "http://proxy.invalid:1234",
            "http_proxy": "http://proxy.invalid:1234",
            "https_proxy": "http://proxy.invalid:1234",
        }
        with patch.dict(os.environ, proxy_environment, clear=False):
            client = OpenRouterClient(
                "dummy",
                network_route_policy=DIRECT_NETWORK_ROUTE_POLICY,
            )
        self.assertFalse(proxy_handlers(client))

    def test_unknown_route_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported network route policy"):
            OpenRouterClient("dummy", network_route_policy="unknown")


if __name__ == "__main__":
    unittest.main()
