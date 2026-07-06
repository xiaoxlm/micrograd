import unittest

from value import Value


class TestValue(unittest.TestCase):
    def test_repr_shows_data(self):
        self.assertEqual(repr(Value(2.0)), "Value(data=2.0)")

    def test_add_returns_value_with_sum(self):
        result = Value(2.0) + Value(3.0)

        self.assertIsInstance(result, Value)
        self.assertEqual(result.data, 5.0)
        
    def test_mul_returns_value_with_product(self):
        result = Value(2.0) * Value(3.0)

        self.assertIsInstance(result, Value)
        self.assertEqual(result.data, 6.0)

# python -m unittest test_value.py
if __name__ == "__main__":
    unittest.main()
