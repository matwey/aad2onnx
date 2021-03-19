from onnx import defs

def get_maximum_opset_supported():
	return defs.onnx_opset_version()
