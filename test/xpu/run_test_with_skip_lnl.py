# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Run the torch-xpu-ops unit tests on LNL, one pytest process per file."""

import sys

from skip_list_common import skip_dict
from skip_list_win import skip_dict as skip_dict_win
from skip_list_win_lnl import skip_dict as skip_dict_win_lnl
from ut_launcher import main, merge_skip_dicts

if __name__ == "__main__":
    overlays = [skip_dict_win, skip_dict_win_lnl] if sys.platform == "win32" else []
    sys.exit(main(merge_skip_dicts(skip_dict, *overlays), __doc__))
