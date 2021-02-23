import os
import subprocess
from typing import Optional
import asyncio
import threading
import abc
import sys


class ExecutableClass(abc.ABC):
    @abc.abstractmethod
    def run(self, args, block=True) -> Optional[int]:
        """
		run exe with supplied command line arguments

		:param args:
		:param block:
		:return: the return code if block is true
		"""
        pass

    @abc.abstractmethod
    async def pipe_output_async(self, file_path=None) -> str:
        """
		Sets up the pipe options.

		Note: The data read is buffered in memory all at once, so do not use this method if the data size is large or
		unlimited.

		:param file_path: file to output to, can be None
		:return: the piped output will be returned
		"""
        pass

    @abc.abstractmethod
    def pipe_output(self, file_path=None, block=False) -> Optional[str]:
        """
		Sets up the pipe options.

		Note: The data read is buffered in memory all at once, so do not use this method if the data size is large or
		unlimited.

		:param file_path: file to output to, can be None
		:param block:
		:return: if block==True on this function or on run then the piped output will be returned
		"""
        pass

    @staticmethod
    @abc.abstractmethod
    def validate_exe_path(exe_path) -> bool:
        """
		Validates whether `exe_path` exists.

		:param exe_path:
		:return: bool
		"""
        pass


class Executable(ExecutableClass):
    """
	A wrapper around Popen
	"""
    __exe = None
    __shell_cmd = False
    __popen_proc = None

    # I don't know if this can be done on windows with out calling the function.
    # At least this is the case if we want the user to be able to run cmd commands.
    @staticmethod
    def validate_exe_path(exe_path):
        """
		Validates that the dentry exists.

		:param exe_path:
		:return: bool
		"""
        if os.path.exists(exe_path):
            return True
        else:
            proc = subprocess.Popen("where {}".format(exe_path), shell=True, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE)
            out, _ = proc.communicate()
            # It should just print out the name of the command,file, or path (or exactly what you gave it) if it exists.
            return exe_path in out.decode('utf-8')

    def __init__(self, exe_path, shell_cmd=False):
        """
		init

		:param exe_path:
		"""
        self.__shell_cmd = shell_cmd
        if not self.__shell_cmd:
            assert self.validate_exe_path(exe_path), 'Make sure the exe path is accessible though cmd.'
        self.__exe = exe_path

    def run(self, args, block=True) -> Optional[int]:
        """
		run exe with supplied command line arguments

		:param args:
		:param block:
		:return: the return code if block is true
		"""
        command = [self.__exe] + args
        command = ' '.join(command)
        self.__popen_proc = subprocess.Popen(command,
                                             shell=self.__shell_cmd,
                                             bufsize=1,
                                             universal_newlines=True,
                                             stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT)

        if block:
            print(f"Executing {command}")
            for c in iter(lambda: self.__popen_proc.stdout.read(1), ''):  # replace '' with b'' for Python 3
                sys.stdout.write(c)
            self.__popen_proc.wait()

    def __wait_for_complete_and_communicate(self, file_path):
        wait_res = self.__popen_proc.wait()
        if wait_res is not None:
            out, err = self.__popen_proc.communicate()
            data = 'out:\n' + out.decode('utf-8') \
                   + (('\nerr:\n' + err.decode('utf-8')) if err is not None else '')
            if file_path is not None:
                with open(file_path, 'w') as f:
                    f.write(data)
                pass
            return wait_res, data

    async def pipe_output_async(self, file_path=None) -> str:
        """
		Sets up the pipe options.

		Note: The data read is buffered in memory all at once, so do not use this method if the data size is large or
		unlimited.

		:param file_path: file to output to, can be None
		:return: the piped output will be returned
		"""

        # TODO: maybe make async version of this function using `asyncio.create_task(wait_for_complete())`
        async def wait_for_complete():
            return self.__wait_for_complete_and_communicate(file_path)

        # can we pipe the output of the tool to a file (eventually log file)
        if self.__popen_proc is not None:
            return await asyncio.create_task(wait_for_complete())
        else:
            return None

    def pipe_output(self, file_path=None, block=False):
        """
		Sets up the pipe options.

		Note: The data read is buffered in memory all at once, so do not use this method if the data size is large or
		unlimited.

		:param file_path: file to output to, can be None
		:param block:
		:return: if block==True on this function or on run then the piped output will be returned
		"""
        # can we pipe the output of the tool to a file (eventually log file)
        if self.__popen_proc is not None:
            if block or self.__popen_proc.poll() is not None:
                return self.__wait_for_complete_and_communicate(file_path)
            else:
                threading.Thread(target=self.__wait_for_complete_and_communicate, args=file_path)
                return None
        else:
            return None
