# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import sys

from run_utils import launch_test
from skip_list_common import skip_dict
from skip_list_win import skip_dict as skip_dict_win
from skip_list_win_lnl import skip_dict as skip_dict_win_lnl

IS_WINDOWS = sys.platform == "win32"

fail_test: list[str] = []
for key in skip_dict:
    skip_list = list(skip_dict[key] or ())
    if IS_WINDOWS:
        skip_list += skip_dict_win.get(key, ()) + skip_dict_win_lnl.get(key, ())
    result: int = launch_test(key, skip_list)
    if result:
        fail_test.append(key)

if fail_test:
    print(",".join(fail_test) + " have failures")
    sys.exit(1)
print("All tests passed!")
