# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Run the torch-xpu-ops unit tests on MTL, one pytest process per file."""

import sys

from skip_list_common import skip_dict
from skip_list_mtl import skip_dict as skip_dict_mtl
from skip_list_win import skip_dict as skip_dict_win
from ut_launcher import main, merge_skip_dicts

if __name__ == "__main__":
    overlays = [skip_dict_mtl]
    if sys.platform == "win32":
        overlays.append(skip_dict_win)
    sys.exit(main(merge_skip_dicts(skip_dict, *overlays), __doc__))
