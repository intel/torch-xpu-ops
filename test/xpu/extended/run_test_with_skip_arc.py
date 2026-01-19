# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import os
import sys

from skip_list_arc import skip_dict as skip_dict_specifical
from skip_list_common import skip_dict
from skip_list_win import skip_dict as skip_dict_win
from skip_list_win_arc import skip_dict as skip_dict_win_arc

IS_WINDOWS = sys.platform == "win32"

skip_list = skip_dict["test_ops_xpu.py"] + skip_dict_specifical["test_ops_xpu.py"]
if IS_WINDOWS:
    skip_list += skip_dict_win["test_ops_xpu.py"] + skip_dict_win_arc["test_ops_xpu.py"]

skip_options = ' -k "not ' + skip_list[0]
for skip_case in skip_list[1:]:
    skip_option = " and not " + skip_case
    skip_options += skip_option
skip_options += '"'

os.environ["PYTORCH_TEST_WITH_SLOW"] = "1"
test_command = "pytest -v test_ops_xpu.py"
test_command += skip_options
res = os.system(test_command)
sys.exit(res)
