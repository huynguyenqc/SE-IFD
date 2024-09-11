# This is a conventional implementation of source code snapshot
# There may be some tricky cases that have not been covered in 
# this implementation.
import os
import shutil
from typing import List


def snapshot(dst_path: str, source_packages: List[str]) -> None:
    dst_abs_path = os.path.abspath(dst_path)

    assert os.path.isdir(dst_abs_path), 'The destination must be a directory!'
    if not dst_abs_path.endswith('/'):
        dst_abs_path += '/'

    current_dir = os.path.abspath(os.getcwd())
    for item in source_packages:
        item_full_path = os.path.join(current_dir, item)
        assert not dst_abs_path.startswith(item_full_path), \
            'It is not valid to copy a folder into itself!'
    
    for item in source_packages:
        item_full_path = os.path.join(current_dir, item)
        target_full_path = os.path.join(dst_abs_path, item)
        
        if os.path.isfile(item_full_path):
            shutil.copyfile(src=item_full_path, dst=target_full_path, follow_symlinks=False)
        else:
            shutil.copytree(src=item_full_path, dst=target_full_path,symlinks=False,
                            ignore=shutil.ignore_patterns('*.pyc', '*.ipynb'))