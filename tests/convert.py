from ad_examples.aad.aad_globals import get_aad_option_list, AAD_IFOREST, AadOpts
from ad_examples.aad.aad_support import get_aad_model
from aad2onnx import convert, to_onnx
import onnx
import onnxruntime as rt
import numpy as np
import unittest

def createAADOpts():
	rng = np.random.RandomState(42)

	parser = get_aad_option_list()
	args = parser.parse_args([])
	args.keep = None
	args.sparsity = np.nan
	args.detector_type = AAD_IFOREST

	opts = AadOpts(args)

	return opts, rng

class ConvertTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls._opts = createAADOpts()

	def createAADModel(self, x):
		model = get_aad_model(x, *self._opts)
		model.fit(x)
		model.init_weights()

		return model

	def evaluate_onnx(self, o, data):
		sess = rt.InferenceSession(o.SerializeToString())
		input_name = sess.get_inputs()[0].name
		label_name = 'score'
		scores_onnx = sess.run([label_name], {input_name: data.astype(np.float32)})[0].reshape(-1)

		return scores_onnx

	def test_convert1(self):
		x = np.random.normal(size=(20, 4)).astype(np.float32)

		model = self.createAADModel(x)
		scores_model = model.get_score(model.transform_to_ensemble_features(x)).astype(np.float32)

		o = to_onnx(model, x)
		onnx.checker.check_model(o)
		scores_onnx = self.evaluate_onnx(o, x)

		np.testing.assert_allclose(scores_onnx, scores_model, 1e-5)
