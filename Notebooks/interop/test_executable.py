from unittest import TestCase
from interop.executable import Executable


class TestExecutable(TestCase):
    def test_run(self):
        # exe = Executable('echo', shell_cmd=True)
        # exe.run(args=['test', 'test2'])
        # ret = exe.pipe_output(block=True)[1]
        #
        # assert 'test' in ret
        # assert 'test2' in ret

        exe = Executable('echo', shell_cmd=True)
        exe.run(args=['test3', 'test4'], block=True)
        # ret = exe.pipe_output()[1]

        # assert 'test3' in ret
        # assert 'test4' in ret

        # t0 = time.time()
        # exe = Executable('timeout')
        # exe.run(args=['2'], block=True)
        # test = exe.pipe_output()
        #
        # assert time.time()-t0 > 1.5

    def test_pipe_output(self):
        pass
