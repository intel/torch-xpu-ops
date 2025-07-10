import csv
import subprocess
import argparse
import json

from smolagents import CodeAgent, HfApiModel, tool, OpenAIServerModel


parser = argparse.ArgumentParser(
                    prog='get_guilty_commit',
                    description='Get the guilty commit',
                    epilog='')
parser.add_argument('--good_commit', type=str, help='The last good commit')
args = parser.parse_args()


OPENAI_API_KEY = "EMPTY"
model = OpenAIServerModel(
    #model_id="Qwen/Qwen2.5-72B-Instruct",
    #model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    model_id="meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://skyrex.jf.intel.com:8000/v1",
    temperature=1,
    api_key=OPENAI_API_KEY,
)

triage_state = {}
triage_state['classify'] = ""
triage_state['random_issue'] = ""
triage_state['guilty_commit'] = ""


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
        base_test_file = "test/" + "/".join(test_class.split('.')[4:-1]).replace('_xpu', '') + '.py'
        test_file = "/".join(test_class.split('.')[0:-1]) + '.py'

        result = row[3].strip()
        err_msg = row[4]
        trace_back = row[5]
        if err_msg in err_msgs:
            continue
        else:

            err_msgs.append(err_msg)

        failure_info = f"Torch-xpu-ops unit test class {test_class} test case {test_case} returned failure with error message: {err_msg}. The test file is {test_file} and as a wrapper file it calls code in the base test file {base_test_file}.\n"
 
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

        # guilty commit -2
        message = f"""
        You are an expert QA, the torch-xpu-ops UT passed on good commit {args.good_commit} but failed on current commit.
        1. Please run  git log to extract the current commit with command
                git log -1 --pretty=format:'%H'
           then the commit range is {args.good_commit}...<current_commit>
        2. Then run the following commands to get a list of commits in the commit range:\n
                git log --pretty=format:'%H' <commit range>
        Please return the answer with a beuitified json format with 'commit_range' as key and seperate the commits wtih ','

        """

        current_commit = run_shell_command('git log -1 --pretty=format:"%H"')
        print(f"The current commit is {current_commit}")
        commit_range = f"{args.good_commit}...{current_commit}"
        commits = run_shell_command(f'git log --pretty=format:"%H" {commit_range}')

        triage_state.update({'commit_range': commits, })
        print(triage_state)
        

        for commit in triage_state['commit_range'].split('\n'):
            print(f"\n\n### check commit {commit}\n")
            message = f"""
            Get the git show result of the commit {commit} with command
                    git show {commit} --function-context

            Then check the output to see whether the commit updated the the code of function {triage_state["classify"]["original_test_case"]} or updated the implementation of the operation {triage_state["classify"]["torch_op"]}.
            If the commit has related updates, return 'Yes' and provide an evidence in a sentence, then summary this commit, otherwise return a string 'No'.
            """

            try:
                output = shell_agent_search.run(
                    message,
                )
            except:
                try:
                    message = f"""
                    Get the git show result of the commit {commit} with command
                            git show {commit} -U 10

                    Then check the output to see whether the commit updated the the code of function {triage_state["classify"]["original_test_case"]} or updated the implementation of the operation {triage_state["classify"]["torch_op"]}.
                    If the commit has related updates, return 'Yes' and provide an evidence in a sentence, then summary this commit, otherwise return a string 'No'.
                    """
                    output = shell_agent_search.run(
                        message,
                    )
                except:
                    print("Skip this commit {commit} as it is too long")

            if 'Yes' in output:
                print(f"### Found a guilty commit! Test case {triage_state['classify']['original_test_case']} failures could due to commit {commit} : {output}\n")
                triage_state.update({commit: output, })


        print("### Result for case {test_case} ###\n")
        print(triage_state)
