import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from camera import main


class CameraStartupTests(unittest.TestCase):
    def test_motion_only_does_not_require_a_model(self):
        with patch("camera.run_camera") as run:
            main(["--motion-only", "--camera-index", "2", "--model", "/missing/model.pt"])
        run.assert_called_once_with(camera_index=2)

    def test_missing_labels_reported_before_inference_import_or_camera(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "AlexNet.onnx"
            model.touch()
            error_output = io.StringIO()
            with patch("camera.run_camera") as run, contextlib.redirect_stderr(error_output):
                with self.assertRaises(SystemExit) as error:
                    main(["--model", str(model)])
            self.assertEqual(error.exception.code, 1)
            self.assertIn("Label-Datei fehlt", error_output.getvalue())
            self.assertIn("--motion-only", error_output.getvalue())
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
