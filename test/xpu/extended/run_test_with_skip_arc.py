import os
import sys
from skip_list_common import skip_dict
from skip_list_arc import skip_dict as skip_dict_specifical

skip_list = skip_dict["test_ops_xpu.py"] + skip_dict_specifical["test_ops_xpu.py"]

skip_options = " -k 'not " + skip_list[0]
for skip_case in skip_list[1:]:
    skip_option = " and not " + skip_case
    skip_options += skip_option
skip_options += "'"

test_command = "PYTORCH_TEST_WITH_SLOW=1 pytest -v test_ops_xpu.py"
test_command += skip_options

res = os.system(test_command)
exit_code = os.WEXITSTATUS(res)
sys.exit(exit_code)
