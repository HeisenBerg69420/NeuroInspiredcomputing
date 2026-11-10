import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from test import OnnxClassifier, collect_images, main
from camera import classify_boxes


class OnnxInferenceTests(unittest.TestCase):
    def test_rgb_preprocessing_predictions_cli_and_label_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / 'colors.onnx'
            graph = helper.make_graph(
                [helper.make_node('ReduceMean', ['images', 'axes'], ['average'], keepdims=0),
                 helper.make_node('MatMul', ['average', 'weights'], ['logits'])],
                'color_classifier',
                [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 8, 8])],
                [helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1, 2])],
                initializer=[numpy_helper.from_array(np.array([2, 3], dtype=np.int64), 'axes'),
                             numpy_helper.from_array(np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32), 'weights')],
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
            model.ir_version = 10
            onnx.checker.check_model(model)
            onnx.save(model, model_path)
            labels = model_path.with_suffix('.labels.txt')
            labels.write_text('red\ngreen\n')
            Image.new('RGB', (20, 12), (255, 0, 0)).save(root / 'red.png')
            Image.new('RGB', (12, 20), (0, 255, 0)).save(root / 'green.png')
            classifier = OnnxClassifier(model_path)
            # Two camera boxes must be cropped independently and converted BGR -> RGB.
            frame = np.zeros((12, 20, 3), dtype=np.uint8)
            frame[:, :10] = (0, 0, 255)
            frame[:, 10:] = (0, 255, 0)
            original = frame.copy()
            predictions = classify_boxes(
                frame, [(-2, 0, 12, 12), (10, 0, 20, 12), (30, 0, 2, 2)], classifier
            )
            self.assertEqual([label for _, label, _ in predictions], ['red', 'green'])
            self.assertEqual([box for box, _, _ in predictions], [(0, 0, 10, 12), (10, 0, 10, 12)])
            self.assertTrue(all(score > 0.8 for _, _, score in predictions))
            np.testing.assert_array_equal(frame, original)
            self.assertEqual(classify_boxes(frame, [], classifier), [])
            with self.assertRaises(ValueError):
                OnnxClassifier(model_path, imagenet=True)

            # The pretrained preset must center-crop, rather than squash the red borders.
            model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 224
            model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 224
            pretrained_path = root / 'imagenet_preprocessing.onnx'
            onnx.save(model, pretrained_path)
            pretrained_path.with_suffix('.labels.txt').write_text('red\ngreen\n')
            imagenet_classifier = OnnxClassifier(pretrained_path, imagenet=True)
            bordered = np.zeros((256, 512, 3), dtype=np.uint8)
            bordered[:] = (255, 0, 0)
            bordered[:, 144:368] = (0, 255, 0)
            self.assertEqual(imagenet_classifier.predict_image(Image.fromarray(bordered))[0][0], 'green')
            np.testing.assert_allclose(imagenet_classifier.mean, [0.485, 0.456, 0.406])
            np.testing.assert_allclose(imagenet_classifier.std, [0.229, 0.224, 0.225])
            for label in ('red', 'green'):
                predictions = classifier.predict(root / f'{label}.png', top_k=2)
                self.assertEqual(predictions[0][0], label)
                self.assertGreater(predictions[0][1], 0.8)
                self.assertAlmostEqual(sum(score for _, score in predictions), 1.0)
            self.assertEqual(len(collect_images([root])), 2)
            with self.assertRaises(ValueError):
                OnnxClassifier(model_path, image_size=16)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(['--model', str(model_path), str(root)])
            self.assertIn('red.png: red', output.getvalue())
            self.assertIn('green.png: green', output.getvalue())
            labels.write_text('wrong\n')
            with self.assertRaises(ValueError):
                OnnxClassifier(model_path).predict(root / 'red.png')
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(['--model', str(root / 'missing.onnx'), str(root)])
            self.assertEqual(error.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
