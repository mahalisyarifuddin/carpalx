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

    def test_keyboard_save_and_reload(self):
        import tempfile
        conf_file = 'etc/carpalx.conf'
        if not os.path.exists(conf_file):
            conf_file = os.path.join(os.path.dirname(__file__), '..', 'etc', 'carpalx.conf')

        app = Carpalx(conf_file)
        app.load_keyboard()
        corpus_path = os.path.join(os.path.dirname(__file__), 'test_corpus.txt')
        app.config['corpus'] = corpus_path
        app.load_triads()

        effort_orig = app.keyboard.calculate_effort(app.triads)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as tmp:
            tmp_path = tmp.name

        try:
            app.keyboard.save(tmp_path)

            from carpalx import Keyboard
            reloaded_kb = Keyboard(tmp_path, app.config)

            self.assertEqual(len(reloaded_kb.keys[0]), 13, "Saved and reloaded layout should keep 13 keys on row 1")
            effort_reloaded = reloaded_kb.calculate_effort(app.triads)
            self.assertAlmostEqual(effort_reloaded, effort_orig, places=5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_corpus_triads_overlap_setting(self):
        import tempfile
        conf_file = 'etc/carpalx.conf'
        if not os.path.exists(conf_file):
            conf_file = os.path.join(os.path.dirname(__file__), '..', 'etc', 'carpalx.conf')

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_corpus:
            tmp_corpus.write("abcdefgh\n")
            tmp_corpus_path = tmp_corpus.name

        try:
            # Test non-overlapping triads (step = 3) -> "abc", "def"
            app_no = Carpalx(conf_file)
            app_no.config['corpus'] = tmp_corpus_path
            app_no.config['triads_overlap'] = 'no'
            app_no.load_keyboard()
            app_no.load_triads()
            self.assertEqual(sum(app_no.triads.values()), 2)
            self.assertIn('abc', app_no.triads)
            self.assertIn('def', app_no.triads)

            # Test overlapping triads (step = 1) -> "abc", "bcd", "cde", "def", "efg", "fgh"
            app_yes = Carpalx(conf_file)
            app_yes.config['corpus'] = tmp_corpus_path
            app_yes.config['triads_overlap'] = 'yes'
            app_yes.load_keyboard()
            app_yes.load_triads()
            self.assertEqual(sum(app_yes.triads.values()), 6)
            self.assertIn('bcd', app_yes.triads)
        finally:
            if os.path.exists(tmp_corpus_path):
                os.remove(tmp_corpus_path)

if __name__ == '__main__':
    unittest.main()
