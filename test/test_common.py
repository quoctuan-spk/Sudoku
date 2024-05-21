# python -m unittest test_common.py
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import common 

class TestCommon(unittest.TestCase):
    def test_list_image_path_normal(self):
        data_path = "images"
        result = common.list_image_path(data_path)
        self.assertIn('images/1.png',result)

    def test_list_image_path_abnormal(self):
        data_path = "fake"
        with self.assertRaises(SystemExit) as cm:
            result = common.list_image_path(data_path)
        self.assertEqual(cm.exception.code, 1)
        
def main():
    unittest.main()
    
if __name__ == "__main__":
    main()
    