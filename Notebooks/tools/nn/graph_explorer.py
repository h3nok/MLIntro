import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

import networkx as nx
from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary

from pprint import pprint

tf.disable_v2_behavior()


class NeuralNetworkExplorer:
	_graph_loc = None
	_graph = None
	_nx_graph = None
	_writer = None

	def __init__(self, graph):
		# this will take at least the .pb vinet file and do all sorts helpful things
		# for instance, plot the graph and save it as an image
		# explore the internal structure of the graph, for instance, the number of outputs
		# number of layers
		# etc etc
		# it will be simple at first.
		self._graph_loc = graph

	def _load_graph(self, file_name):
		"""
		Loads graph from pb file
		@param file_name: pb file location
		"""
		if not os.path.exists(file_name):
			raise ValueError("Specified graph file doesn't exist")
		with open(file_name, 'rb') as f:
			content = f.read()

		graph_def = tf.GraphDef()
		graph_def.ParseFromString(content)

		with tf.Graph().as_default() as graph:
			tf.import_graph_def(graph_def, name='')

		self._graph = graph
		self._nx_graph = None
		self._graph_loc = file_name

	def _load_nx_graph(self):
		"""
		Loads nx graph from loaded pb graph
		"""
		self._nx_graph = nx.Graph()
		for n in self.get_nodes(print_nodes=False):
			self._nx_graph.add_node(n.name, node_info=n)
			for input_name in n.input:
				assert self._nx_graph.has_node(input_name)
				self._nx_graph.add_edge(input_name, n.name)

	def rename_output(self, output_graph, output_nodes, output_name='output'):
		"""
		Renames output node. This would also work for any renaming
		@param output_graph: file location for resulting pb file
		@param output_nodes: output node to change
		@param output_name: name to change the output nodes to.
		"""
		if self._graph is None:
			self._load_graph(self._graph_loc)
		graph_def = self._graph.as_graph_def()

		for node in graph_def.node:
			if output_nodes in node.name:
				node.name = output_name
		# Serialize and write to file
		with tf.gfile.GFile(output_graph, "wb") as f:
			f.write(graph_def.SerializeToString())
		print("%d ops in the final graph." % len(graph_def.node))

	def get_nodes(self, print_nodes=True):
		"""
		Returns list of nodes
		@param print_nodes: print nodes to terminal
		"""
		if self._graph is None:
			self._load_graph(self._graph_loc)

		graph_def = self._graph.as_graph_def(add_shapes=True)

		nodes = [n for n in graph_def.node]
		if print_nodes:
			for n in nodes:
				pprint(n)
		return nodes

	def get_ops(self):
		"""
		Get list of operations
		"""
		if self._graph is None:
			self._load_graph(self._graph_loc)
		return self._graph.get_operations()

	def print_ops(self, input_only=False):
		"""
		Print operations. Splits into operations without input and operations with inputs
		@param input_only:
		"""
		if self._graph is None:
			self._load_graph(self._graph_loc)
			self._load_nx_graph()

		ops = self.get_ops()
		print("Number of operations: {}".format(len(ops)))
		if not input_only:
			for op in ops:
				print("{} => {}".format(op.name, op.values()))

		print()
		print('Sources (operations without inputs):')
		for op in ops:
			if len(op.inputs) > 0:
				continue
			print('- {0}'.format(op.name))

		print()
		print('Operation inputs:')
		for op in ops:
			if len(op.inputs) == 0:
				continue
			print('- {0:20}'.format(op.name))
		print('  {0}'.format(', '.join(i.name for i in op.inputs)))

		print()
		print('Tensors:')
		for op in ops:
			for out in op.outputs:
				print('- {0:20} {1:10} "{2}"'.format(str(out.shape), out.dtype.name, out.name))

	@staticmethod
	def list_to_file(l, output_file):
		"""
		This will write any of the lists created by other function and save it to a file
		@param l: list to save
		@param output_file: file to save list to
		"""
		with open(output_file, "w") as f:
			f.write("Number of nodes: %d" % len(l))
			f.write('\n')
			for item in l:
				f.write(item)
				f.write('\n')
		f.close()

	def save_as_image(self, save_to='', show=False):
		""" Not implemented. Im not sure how to approach this
		TODO: finish this"""
		if self._graph is None:
			self._load_graph(self._graph_loc)
			self._load_nx_graph()
		raise NotImplementedError()

	def import_to_tensorboard(self, log_dir):
		"""
		View an imported protobuf model (`.pb` file) as a graph in Tensorboard.
		This would be great but sadly it doesnt work
		@param log_dir: The location for the Tensorboard log to begin visualization from.
		"""
		with session.Session(graph=ops.Graph()) as sess:
			with gfile.FastGFile(self._graph_loc, "rb") as f:
				graph_def = graph_pb2.GraphDef()
				graph_def.ParseFromString(f.read())
				importer.import_graph_def(graph_def)

			pb_visual_writer = summary.FileWriter(log_dir)
			pb_visual_writer.add_graph(sess.graph)
			print("Model Imported. Visualize by running: tensorboard --logdir={}".format(log_dir))
