from setuptools import setup, find_packages

name="aad2onnx"
version="0.1.4"
test_suite="tests"

setup(name=name,
	version=version,
	description='Python module to convert AAD model to ONNX format',
	url='http://github.com/matwey/aad2onnx',
	author='Matwey V. Kornilov',
	author_email='matwey.kornilov@gmail.com',
	packages=find_packages(exclude=(test_suite,)),
	test_suite=test_suite,
	license='MIT',
	install_requires=[
		'onnx',
		'onnxconverter-common >= 1.8.0',
		'onnxruntime',
		'skl2onnx',
	],
	classifiers=[
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
	],
	zip_safe=False)
