#!/usr/bin/env python3
"""
Legacy shim: ``add_methods`` is deprecated (no-op). Assert the real module exposes APIs.
"""

import unittest


class TestAddMethodsDeprecated(unittest.TestCase):
    def test_memory_enhancer_has_expected_methods(self):
        from backend.memory_enhancer import MemoryEnhancer

        self.assertTrue(hasattr(MemoryEnhancer, "background_process"))
        self.assertTrue(hasattr(MemoryEnhancer, "get_statistics"))


if __name__ == "__main__":
    unittest.main()
