import subprocess
from typing import Optional
import asyncio
import threading
from interop.executable import ExecutableClass


def to_wsl_path(win_path):
    """
    Takes a path to a dentry and converts it to something that wsl can use.
    Can be an absolute path or a relative path.

    :param win_path: windows path (can be from C:/ or some other drive or a relative path)
    :return: wsl path starting at /mnt/c/ or some other drive
    """
    win_path = win_path.strip("'").strip('"')
    if ':' in win_path:
        splits = win_path.split(':\\')
        assert len(splits) == 2
        root = '/mnt/{}/'.format(splits[0].lower())
        win_path = root + splits[1]

    return win_path.replace('\\', '/')


class WslExecutable(ExecutableClass):
    """
    A wrapper around Popen and the windows wsl.exe cmd command
    """
    __exe = None
    __popen_proc = None

    @staticmethod
    def validate_exe_path(exe_path):
        """
        Validates that the command or dentry exists.

        :param exe_path: wsl format
        :return: bool
        """
        proc = subprocess.Popen("wsl command -v {}".format(exe_path), shell=True, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        out, _ = proc.communicate()
        # It should just print out the name of the command,file, or path (or exactly what you gave it) if it exists.
        return exe_path in out.decode('utf-8')

    def __init__(self, exe_path):
        """
        init

        :param exe_path: in wsl format
        """
        assert self.validate_exe_path('echo'), 'Make sure you have WSL set up.'
        assert self.validate_exe_path(exe_path), 'Make sure the exe path is accessible though the cmd wsl command ' \
                                                 'i.e. `wsl <exe_path>` works from cmd.'
        self.__exe = exe_path

    def run(self, args, block=True) -> Optional[int]:
        """
        run exe with supplied command line arguments

        :param args: any paths must be in WSL format. Use `to_wsl_path`.
        :param block:
        :return: the return code if block is true
        """
        # It seems that Popen doesn't get the stderr from wsl so we add 2>&1 to pipe the stderr to stdout
        command = ['wsl', self.__exe] + args + ['2>&1']
        command_str = ''
        for c in command:
            command_str += c + ' '

        self.__popen_proc = subprocess.Popen(command_str, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        if block:
            return self.__popen_proc.wait()
        else:
            return None

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

    async def pipe_output_async(self, file_path=None):
        """
        Sets up the pipe options.

        Note: The data read is buffered in memory all at once, so do not use this method if the data size is large or
        unlimited.

        :param file_path: file to output to (windows path), can be None
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

        :param file_path: file to output to (windows format), can be None
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


