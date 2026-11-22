"""Route old evaluation commands without changing the ordinary infer API."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from contextdepth.cli import main


class CLICompatibilityTests(unittest.TestCase):
    def assert_evaluation_route(self, argv, forwarded=None):
        with patch("evaluation.cli.main", return_value=17) as evaluate, \
                patch("contextdepth.cli.parser", side_effect=AssertionError("ordinary parser called")):
            self.assertEqual(main(argv), 17)
            evaluate.assert_called_once_with(argv if forwarded is None else forwarded)

    def test_legacy_reference_manifest_routes_with_both_option_forms(self):
        for option in (["--reference-bank", "reference_bank.json"],
                       ["--reference-bank=reference_bank.json"]):
            with self.subTest(option=option):
                self.assert_evaluation_route(["infer", "--image", "q.png", *option, "--output", "out"])

    def test_legacy_model_option_routes_with_both_option_forms(self):
        for option in (["--models", "moge2"], ["--models=moge2"]):
            with self.subTest(option=option):
                self.assert_evaluation_route(["infer", "--image", "q.png", *option, "--output", "out"])

    def test_existing_evaluation_routes_preserve_arguments(self):
        for argv in (["predict", "--manifest", "inputs.json"],
                     ["generate", "--manifest", "inputs.json"],
                     ["infer", "--prepare-only", "--image", "q.png"]):
            with self.subTest(argv=argv):
                self.assert_evaluation_route(argv)

    def test_explicit_eval_prefix_is_removed(self):
        self.assert_evaluation_route(["eval", "infer", "--image", "q.png"],
                                     ["infer", "--image", "q.png"])

    def test_ordinary_infer_and_diode_use_the_public_pipeline(self):
        cases = [([], None), (["--reference-bank", "diode"], "diode"),
                 (["--reference-bank=diode"], "diode")]
        with tempfile.TemporaryDirectory() as temporary:
            for option, bank in cases:
                with self.subTest(option=option), patch("evaluation.cli.main") as evaluate, \
                        patch("contextdepth.pipeline.ContextDepthPipeline") as factory, patch("builtins.print"):
                    argv = ["infer", "--image", "q.png", "--output", temporary, *option]
                    if bank:
                        argv += ["--diode-root", "data/DIODE"]
                    self.assertEqual(main(argv), 0)
                    evaluate.assert_not_called()
                    self.assertEqual(factory.call_args.kwargs["reference_bank"], bank)
                    self.assertEqual(factory.call_args.kwargs["diode_root"], Path("data/DIODE") if bank else None)
                    pipeline = factory.return_value.__enter__.return_value
                    pipeline.assert_called_once_with(Path("q.png"), camera=None)
                    pipeline.return_value.save.assert_called_once_with(Path(temporary))

    def test_incomplete_bank_option_remains_an_ordinary_argument_error(self):
        with patch("evaluation.cli.main") as evaluate, redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["infer", "--image", "q.png", "--output", "out", "--reference-bank"])
            self.assertEqual(raised.exception.code, 2)
            evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
