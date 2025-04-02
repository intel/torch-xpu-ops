import re
import sys
from io import StringIO
import pytest
from skip_list_win_mtl import skip_dict

IS_WINDOWS = sys.platform == "win32"

skip_list = skip_dict["test_xpu.py"]

skip_options = "not " + skip_list[0]
for skip_case in skip_list[1:]:
    skip_option = " and not " + skip_case
    skip_options += skip_option

original_stdout = sys.stdout
sys.stdout = StringIO()

test_command = ["-k", skip_options, "../../../../test/test_xpu.py", "-v"]
res = pytest.main(test_command)

output = sys.stdout.getvalue()
sys.stdout = original_stdout

cleaned_output = re.sub(
    r"\.\.(\/|\\)\.\.(\/|\\)\.\.(\/|\\)\.\.(\/|\\)test(\/|\\)", "", output
)
print(cleaned_output, end="")
sys.exit(res)
