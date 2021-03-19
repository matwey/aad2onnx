from setuptools import find_packages, setup

name="aad2onnx"
version="0.1"
test_suite="tests"

setup(name=name,
	version=version,
	description='Python module to convert AAD model to ONNX format',
	url='http://github.com/matwey/aad2onnx',
	author='Matwey V. Kornilov',
	author_email='matwey.kornilov@gmail.com',
	license='MIT',
	packages=find_packages(exclude=(test_suite,)),
	test_suite=test_suite,
	zip_safe=False)
