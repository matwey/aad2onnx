from ._container import AadModelContainer
from onnxconverter_common.topology import Topology
from onnxconverter_common.data_types import FloatTensorType

from ad_examples.aad.forest_aad_detector import AadForest

def _get_aad_operator_name(model):
	# FIXME: not all possible AAD models are currently supported
	if not isinstance(model, AadForest):
		raise ValueError("No proper operator name found for '%s'" % type(model))

	return "AadForest"

def _parse_aad(scope, model, inputs):
	this_operator = scope.declare_local_operator(_get_aad_operator_name(model), model)
	this_operator.inputs = inputs

	# FIXME: probably another variable is required for anomality label
	score_variable = scope.declare_local_variable('score', FloatTensorType())

	this_operator.outputs.append(score_variable)

	return this_operator.outputs

def parse_aad(model, initial_types=None, target_opset=None,
	custom_conversion_functions=None, custom_shape_calculators=None):

	raw_model_container = AadModelContainer(model)
	topology = Topology(raw_model_container, default_batch_size='None',
		initial_types=initial_types, target_opset=target_opset,
		custom_conversion_functions=custom_conversion_functions,
		custom_shape_calculators=custom_shape_calculators)
	scope = topology.declare_scope('__root__')

	inputs = []
	for var_name, initial_type in initial_types:
		inputs.append(scope.declare_local_variable(var_name, initial_type))

	for variable in inputs:
		raw_model_container.add_input(variable)

	outputs = _parse_aad(scope, model, inputs)

	for variable in outputs:
		raw_model_container.add_output(variable)

	return topology
