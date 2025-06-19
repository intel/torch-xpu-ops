import csv
import subprocess
import argparse
import json

from smolagents import CodeAgent, HfApiModel, tool, OpenAIServerModel


parser = argparse.ArgumentParser(
                    prog='traige',
                    description='Triage UT failure with AI agent',
                    epilog='')
parser.add_argument('--good_commit', type=str, help='The last good commit')
args = parser.parse_args()


OPENAI_API_KEY = "EMPTY"
model = OpenAIServerModel(
    model_id="Qwen/Qwen2.5-72B-Instruct",
    api_base="http://skyrex.jf.intel.com:8000/v1",
    temperature=1,
    api_key=OPENAI_API_KEY,
)

triage_state = {}
triage_state['classify'] = ""
triage_state['random_issue'] = ""
triage_state['guilty_commit'] = ""

def retest(test_file: str, test_case: str) -> str :
    """
    Rerun test and return the output as a string
    Args:
       test_file: The pytest test file
       test_case: The pytest test case name start with 'test_'
    """
    command = f"pytest -v {test_file} -k {test_case}"
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e}"


@tool
def run_shell_command(command: str) -> str:
    """
    Executes a shell command.
    Args:
        command: The shell command to execute.
    """
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e}"


def output_to_dict(output):
   if isinstance(output, dict):
       output_dict = output
   else:
       try:
           import json
           output_dict = json.loads(output)
       except:
           print("The output json is invalid")
           return None
   return output_dict


chat_agent = CodeAgent(tools=[],
                 model=model,
                 add_base_tools=True,
                 additional_authorized_imports=['json', 're', 'subprocess'])

shell_agent = CodeAgent(tools=[run_shell_command],
                 model=model,
                 add_base_tools=False,
                 additional_authorized_imports=['json' 're', 'subprocess'])

shell_agent_search = CodeAgent(tools=[run_shell_command],
                 model=model,
                 add_base_tools=True,
                 additional_authorized_imports=['json' 're', 'subprocess'])



fail_list_path = "ut_failure_list.csv"

with open(fail_list_path, 'r') as csvfile:
    _reader = csv.reader(csvfile, delimiter='|')
    err_msgs = []
    for row in _reader:
        test_class = row[1].strip()
        test_case = row[2].strip()
        base_test_file = "test/" + test_class.split('.')[-2].replace('_xpu', '') + '.py'
        test_file = "/".join(test_class.split('.')[0:-1]) + '.py'

        result = row[3].strip()
        err_msg = "".join(row[4:])
        if err_msg in err_msgs:
            continue
        else:
            err_msgs.append(err_msg)

        failure_info = f"Torch-xpu-ops unit test class {test_class} test case {test_case} returned failure with error message: {err_msg}. The test file is {test_file} and as a wrapper file it calls code in the base test file {base_test_file}.\n"

        #message = f"""
        #You are an expert QA, here we observed an error in torch-xpu-ops UT test
        #{failure_info}

        #1. Please extract the test case name started with 'test_' from above information.
        #2. The test case could be created by parameterizing on the orignal test case, when pramiterization the test cases will be named with the original test case as prefix and with information like torch operation, operation mode, backend, and dtype as post fix and seperated with '_', for example when origianl test name is 'test_compare_cpu', the test case could be 'test_compare_cpu__batch_norm_with_update_xpu_bool'. With the test case name please get the original test case name from file https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/{base_test_file}.
        #3. Please help to check what is the module to test, torch operation, distributed, runtime, inductor or torchbench? 
        #4. Please help to check the dtype of the test.
        #5. If the test is for torch operation, please get the related torch operations or tensor information, the method is:
        #   a. If the test case is parameterized on torch operation, get the torch operation from test case name. 
        #   b. If the test case is not parameterized on torch operation, check the {base_test_file} and traceback to get the torch operation. It is a general torch tensor name without backend ('xpu', 'cuda', 'cpu', 'hpu', etc.) inforamtion in the torch tensor.
        #6. Is the failure an AssertionError, FatalError or RuntimeError or other errors?
        #7. What is the error message?
        #Please list the answer in a beutified json format with key 'base_test_file', 'test_case', 'original_test_case', 'module', 'dtype', 'torch_op', 'error_type', 'error_message'.
        #"""

        message = f"""
You are an expert QA engineer analyzing a failure in the `torch-xpu-ops` unit test. Below is the failure log:

{failure_info}

### Tasks:
1. **Extract the test case name**:
   - Identify the full test case name starting with `test_` from the failure log.

2. **Determine the original test case**:
   - If the test is parameterized (e.g., `test_compare_cpu__batch_norm_with_update_xpu_bool`), extract the base test name (e.g., `test_compare_cpu`).
   - Verify by checking the test file:
     ```
     https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/{base_test_file}
     ```
3. **Identify the test module**:
   - Categorize the test into one of:
     - `torch operation` (core ops)
     - `distributed` (multi-process/device)
     - `runtime` (execution environment)
     - `inductor` (compiler)
     - `torchbench` (benchmarking)

4. **Extract the data type (`dtype`)**:
   - Infer from test name/trace (e.g., `xpu_bool` → `bool`, `f32` → `float32`).

5. **Resolve the torch operation or tensor info**:
   - **If parameterized**: Parse the op name from the test case (e.g., `batch_norm` from `...batch_norm...`), the op name is usually right after the base test name.
   - **Otherwise**: Trace the failure in `{base_test_file}` to find the tensor/op (ignore backends like `xpu/cpu`).

6. **Classify the error type**:
   - `AssertionError` (test condition failed)
   - `RuntimeError` (execution crash)
   - `FatalError` (critical system error)
   - Other (specify).

7. **Extract the error message**:
   - Copy the raw error message (exclude stack traces unless critical).

### Output Format:
Return JSON with the following structure:
```json
{{
  "base_test_file": "path/to/base_test.py",
  "test_file": "path/to/test.py"
  "test_case": "full_parameterized_test_name",
  "original_test_case": "base_test_name",
  "module": "torch operation/distributed/runtime/inductor/torchbench",
  "dtype": "float32/bool/int64/etc",
  "torch_op": "op_name_or_None",
  "error_type": "AssertionError/RuntimeError/etc",
  "error_message": "Raw error text"
}}

```
"""

        output = chat_agent.run(
            message,
        )

        classifications = output_to_dict(output)

        if classifications is None:
            continue

        triage_state.update({'classify': classifications, })

        # Retest
        retest_output = retest(classifications['test_file'], classifications['test_case'])
        message = f"""
        You are an expert QA, here we observed an error in torch-xpu-ops UT test
        {failure_info}
        And we did a retest and got output {retest_output}.
        If the two test output are not for the same error, or one is passed, return 'random_issue', otherwise return 'not_random_issue'. 
        Please list the answer in a beutified json format with key 'randomness'.
        """
        output = chat_agent.run(
            message,
        )

        randomness = output_to_dict(output)
        if randomness is None:
            continue

        triage_state.update({'randomness': randomness, })

        # guilty commit
        message = f"""
        You are an expert QA, the torch-xpu-ops UT passed on good commit {args.good_commit} but failed on current commit.
        1. Please run  git log to extract the current commit with command
                git log -1 --pretty=format:'%H'
           then the commit range is <good_commit>...<current_commit>
        2. Then run the following commands to get a list of commits in the commit range:\n
                git log --pretty=format:'%H' <commit range>
        3. For each commit in the commit range, check the git show result of that commit with command
                git show <commit>
           and check the git show output to see whether the commit has any backend specific update on cuda, hpu or xpu, etc.
        Please return the answer with beutified json format with key "commit" and "evidence", please provide a sentence as evidence.
        """

        output = shell_agent.run(
            message,
        )
        backend_updated = output_to_dict(output)
        triage_state.update({'backend_updated': backend_updated, })


        # guilty commit -2
        message = f"""
        You are an expert QA, the torch-xpu-ops UT passed on good commit {args.good_commit} but failed on current commit.
        1. Please run  git log to extract the current commit with command
                git log -1 --pretty=format:'%H'
           then the commit range is <good_commit>...<current_commit>
        2. Then run the following commands to get a list of commits in the commit range:\n
                git log --pretty=format:'%H' <commit range>
        3. For each commit in the commit range, get the git show result of that commit with command
                git show <commit> --function-context

        Then check the output to see whether the commit updated the the code of function {triage_state["classify"]["original_test_case"]} or updated the implementation of the operation {triage_state["classify"]["torch_op"]}.
        Please return the answer with beutified json format with key "commit" and "evidence, please provide a sentence or a piece of code statement as the "evidence".
        """

        output = shell_agent_search.run(
            message,
        )
        guilty_commit = output_to_dict(output)
        triage_state.update({'possible_guilty_commit': guilty_commit, })

        print("### Result ###\n")
        print(triage_state)
