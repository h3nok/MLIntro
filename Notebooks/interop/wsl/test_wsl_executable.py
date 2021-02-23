import time
from unittest import TestCase
from wsl_executable import WslExecutable, to_wsl_path


class TestWslExecutable(TestCase):
    def test_validate_exe_path(self):
        assert WslExecutable.validate_exe_path('echo')
        assert WslExecutable.validate_exe_path('/bin/echo')
        assert not WslExecutable.validate_exe_path('this/command/does/not/exists')
        assert not WslExecutable.validate_exe_path('this_command_does_not_exists')

    def test_run(self):
        exe = WslExecutable('echo')
        exe.run(args=['test', 'test2'])
        ret = exe.pipe_output(block=True)[1]

        assert 'test' in ret
        assert 'test2' in ret

        exe = WslExecutable('echo')
        exe.run(args=['test3', 'test4'], block=True)
        ret = exe.pipe_output()[1]

        assert 'test3' in ret
        assert 'test4' in ret

        t0 = time.time()
        exe = WslExecutable('sleep')
        exe.run(args=['2'], block=True)

        assert time.time()-t0 > 1.5

    def test_pipe_output(self):
        pass  # TODO:

    def test_to_wsl_path(self):
        test_path1 = r'test\win\path'
        test_path2 = r'C:\test\win\path2'
        test_path3 = r'F:\test\win\path3'
        assert to_wsl_path(test_path1) == test_path1.replace('\\', '/')
        assert to_wsl_path(test_path2) == r'/mnt/c/test/win/path2'
        assert to_wsl_path(test_path3) == r'/mnt/f/test/win/path3'
