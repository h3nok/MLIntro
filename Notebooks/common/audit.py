import os

path_exists = lambda file: os.path.exists(file)
is_file = lambda file: os.path.isfile(file)
is_dir = lambda d: os.path.isdir(d)
endswith = lambda item, ext: item.endswith(ext)
