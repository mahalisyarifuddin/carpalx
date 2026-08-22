import unittest
import os
import sys

# Ensure repo root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from carpalx import Carpalx

class TestPythonLayout(unittest.TestCase):
    def test_qwerty_layout_loading_and_effort(self):
        conf_file = 'etc/carpalx.conf'
        if not os.path.exists(conf_file):
            conf_file = os.path.join(os.path.dirname(__file__), '..', 'etc', 'carpalx.conf')

        app = Carpalx(conf_file)
        app.load_keyboard()

        # Verify row 1 key count and escaped # parsing
        row1 = app.keyboard.keys[0]
        self.assertEqual(len(row1), 13, "Row 1 should contain 13 keys")
        self.assertEqual(row1[3]['lc'], '3')
        self.assertEqual(row1[3]['uc'], '#')

        # Verify total effort on test corpus matches JS/Perl baseline
        corpus_path = os.path.join(os.path.dirname(__file__), 'test_corpus.txt')
        app.config['corpus'] = corpus_path
        app.load_triads()
        effort = app.keyboard.calculate_effort(app.triads)

        self.assertAlmostEqual(effort, 3.527854, places=5)

if __name__ == '__main__':
    unittest.main()
