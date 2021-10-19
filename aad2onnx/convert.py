from uuid import uuid4

import onnx
from .proto import get_maximum_opset_supported
from ._parse import parse_aad
from . import operator_converters
from . import shape_calculators
from onnxconverter_common.topology import convert_topology

def convert(model, name=None, initial_types=None, doc_string='', target_opset=None,
	targeted_onnx=onnx.__version__, custom_conversion_functions=None,
	custom_shape_calculators=None):

	if initial_types is None:
		raise ValueError('Initial types are required. See usage of '
				'convert(...) in aad2onnx.convert for details')

	if name is None:
		name = str(uuid4().hex)

	target_opset = target_opset if target_opset else get_maximum_opset_supported()

	topology = parse_aad(model, initial_types, target_opset, custom_conversion_functions, custom_shape_calculators)
	topology.compile()

	onnx_model = convert_topology(topology, name, doc_string, target_opset, targeted_onnx)
	return onnx_model

def to_onnx(model, X=None, name=None, initial_types=None, target_opset=None,
	targeted_onnx=onnx.__version__, custom_conversion_functions=None,
	custom_shape_calculators=None):

	from skl2onnx.algebra.type_helper import guess_initial_types

	if name is None:
		name = "ONNX(%s)" % model.__class__.__name__
	initial_types = guess_initial_types(X, initial_types)

	return convert(model, name=name, initial_types=initial_types,
		target_opset=target_opset, targeted_onnx=targeted_onnx,
		custom_conversion_functions=custom_conversion_functions,
		custom_shape_calculators=custom_shape_calculators)
