from abc import ABC, abstractmethod
from tools.nn import NeuralNetworkExplorer
from interop.cpp_interop import CPPInterop
import configparser


class ModelBase(ABC):

	@abstractmethod
	def config(self):
		pass

	@abstractmethod
	def customer(self):
		pass

#
# class CandidateModel(ModelBase):
# 	_config = None
# 	_pb_file = None
# 	_graph_explorer: NeuralNetworkExplorer = None
#
# 	def __init__(self, model_config, customer):
# 		# config can be the pb file for now.
# 		# TODO - sproc to fetch config metadata and graph
# 		if '.pb' in model_config:
# 			self._pb_file = model_config
# 		else:
# 			self._config = model_config
# 			# TODO fetch pb file
# 		self._customer = customer
#
# 		if self._pb_file:
# 			self._graph_explorer = NeuralNetworkExplorer(self._config)
#
# 	@property
# 	def customer(self) -> str:
# 		return self._customer
#
# 	@property
# 	def config(self):
# 		return self._config
#
# 	def evaluate(self, tag):
# 		"""
# 		evaluate the config on the supplied tag
# 		@param tag: specified tag
# 		"""
# 		# These are hardcoded for now but will hopefully change
# 		db_ini = r'C:\ProgramData\viNet\config\database.ini'
# 		classify_tool_exe = r'C:\ProgramData\viNet\tools\viNetClassifyTool\viNetClassifyTool.exe'
#
# 		db_config = configparser.ConfigParser()
# 		db_config.read(db_ini)
#
# 		host = db_config.get('postgresql', 'host')
# 		database = db_config.get('postgresql', 'database')
#
# 		print(f"Config: {self._config}")
# 		print(f"Tag: {tag}")
#
# 		command = CPPInterop(classify_tool_exe)
# 		command.build_command(config=self._config, tag=tag, h=host, d=database)
#
# 		command.execute()
#
# 	def upload(self):
# 		# same as add_config
# 		# TODO - Henok, port c++ to python
# 		pass
#
# 	def plot(self, save_as="", show=False):
# 		# call into nn to plot the network and save it as image
# 		self._graph_explorer.save_as_image('log')
