from ad_examples.aad.aad_globals import get_aad_option_list, AAD_IFOREST, AadOpts
from ad_examples.aad.aad_support import get_aad_model
from aad2onnx import convert, to_onnx
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

	def test_convert1(self):
		x = np.random.normal(size=(20, 4))

		model = self.createAADModel(x)

		pass
