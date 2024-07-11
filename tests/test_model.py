import unittest
from src.models.model import get_model_and_tokenizer

class TestModel(unittest.TestCase):
    def test_get_model_and_tokenizer(self):
        model, tokenizer = get_model_and_tokenizer('distilbert-base-uncased')
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)

if __name__ == '__main__':
    unittest.main()

