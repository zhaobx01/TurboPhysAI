import importlib
import unittest


class CatalogTest(unittest.TestCase):
    def test_catalog_can_be_imported(self):
        importlib.import_module("groot_optimization.catalog")


if __name__ == "__main__":
    unittest.main()
