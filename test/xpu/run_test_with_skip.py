# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Run the torch-xpu-ops unit tests of skip_list_common.py, one pytest process per file.

See --help, the launcher lives in ut_launcher.py.
"""

import os
import sys

from skip_list_common import skip_dict
from ut_launcher import main, merge_skip_dicts


def platform_overlays():
    if os.name != "nt":
        return []
    try:
        from windows_skip_dict import skip_dict as win_skip_dict
    except ImportError:
        print("Warning: windows_skip_dict not found, using the default skip dict")
        return []
    return [win_skip_dict]


if __name__ == "__main__":
    sys.exit(
        main(merge_skip_dicts(skip_dict, *platform_overlays()), __doc__.splitlines()[0])
    )
