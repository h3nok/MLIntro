import os
import subprocess


class CPPInterop:
    __cpp_file = None
    _command_str = None

    def __init__(self, file):
        assert os.path.exists(file)
        self.__cpp_file = file
        self._command_str = self.__cpp_file

    def build_command(self, use_equals=False, **kwargs):
        """
        Builds Command string
        @param use_equals: instead of --ab txt it will generate --ab=txt
        @param kwargs: specifies arguments for the command.
        For example with the command test.exe --ab txt --abc "text space" -h 1.1.1.1 -l
        command(ab="txt", abc="text space", h="1.1.1.1", l="").
        If non-str types are provided they will be cast to str
        """
        # loop through arguments
        for k in kwargs:
            v = str(kwargs[k])
            # depending on length of key we will do -- or -
            if len(k) == 1:
                self._command_str += f" -{k}"
            else:
                self._command_str += f" --{k}"

            # add value if needed
            if len(v) > 0:
                if " " in v:
                    v = f'"{v}"'

                # Combine key and value
                if use_equals:
                    self._command_str += f"={v}"
                else:
                    self._command_str += f" {v}"

    @property
    def command(self):
        return self._command_str

    def execute(self):
        """ Execute command str """
        print(f"Executing : {self._command_str}")
        subprocess.call(self._command_str)
