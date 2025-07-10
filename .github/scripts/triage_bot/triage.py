import csv
import subprocess
import argparse
import json

from smolagents import CodeAgent, HfApiModel, tool, OpenAIServerModel

#model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

parser = argparse.ArgumentParser(
                    prog='traige',
                    description='Triage UT failure with AI agent',
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

@tool
def open_local_file(path: str, max_length: int) -> str:
    """
    Open the local file and truncate if the length is less than max_length
    Args:
        path: The path of the local file
        max_length: If larger than max_length, truncate the file
    """
    try:
        with open(path, 'r') as file:
            content = file.read()

            def _truncate_content(content: str, max_length: int) -> str:
                 if len(content) <= max_length:
                     return content
                 return (
                     content[: max_length // 2]
                     + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
                     + content[-max_length // 2 :]
                 )

            return _truncate_content(content, max_length)
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
        return f"Error: The file {path} was not found."


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


#retest_agent = CodeAgent(tools=[retest],
#                 model=model,
#                 add_base_tools=False,
#                 additional_authorized_imports=['json', 'subprocess'])



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

        #message = f"""
        #You are an expert QA, here we observed an error in torch-xpu-ops UT test
        #{failure_info}
        #The test is a wrapper of the original test file {base_test_file}".
        #1. First create a local directory retest if it does not exists
        #2. download https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/{base_test_file} to local disk as {base_test_file}
        #3. Create a new {triage_state['classify']['original_test_case']} accroding to the {triage_state['classify']['original_test_case']} of the local file,  before each statement of {triage_state['classify']['original_test_case']} add a print to output the shape of each auguments. 
        #4. Remove the {triage_state['classify']['original_test_case']} from the local file and replcae it with the new function.
        #"""

#        message = f"""
#You are an expert QA engineer helping to debug a torch-xpu-ops unit test failure. Here are the details:
#
#Test Failure Information:
#{failure_info}
#
#The test is a wrapper of the original test file located at:
#https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/{base_test_file}
#
#Please perform the following steps to help diagnose the issue:
#
#1. Create a local directory named 'retest' if it doesn't already exist
#2. Download the original test file from:
#   https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/{base_test_file}
#   and save it locally with the same filename: {base_test_file}
#3. Analyze the test case '{triage_state['classify']['original_test_case']}' from the local file
#4. Create a modified version of this test case that:
#   - Preserves all original functionality
#   - Adds print statements before each operation to output:
#     * The shape of all input tensors
#     * The dtype of all input tensors
#     * The value of all input tensors
#     * Any other relevant tensor metadata
#From the traceback we could find the assertion statement in the test case, the goal is to create a version of the test that will help us understand the tensor shapes and types at each step of execution from the very beginning of the test to the assertion statement, which should help diagnose where the failure is occurring to trigger the assertion.
#
#Please pay special attention to:
#- Any tensor operations in the test case
#- The expected vs. actual shapes at each step
#- Any type conversions or special conditions
#- Edge cases that might be triggering the failure
#
#Please output the new test case for me and don't update the file.
#"""
##5. Replace the original test case in the local file with this instrumented version
##6. Ensure the modified file maintains all imports and other test cases
#
#
#        import pdb
#        pdb.set_trace()
#        output = chat_agent.run(
#            message,
#        )
#
#        print(output)
 
        ## Retest
        #retest_output = retest(classifications['test_file'], classifications['test_case'])
        #message = f"""
        #You are an expert QA, here we observed an error in torch-xpu-ops UT test
        #{failure_info}
        #And we did a retest and got output {retest_output}.
        #If the two test output are not for the same error, or one is passed, return 'random_issue', otherwise return 'not_random_issue'. 
        #Please list the answer in a beutified json format with key 'randomness'.
        #"""
        #output = chat_agent.run(
        #    message,
        #)

        #randomness = output_to_dict(output)
        #if randomness is None:
        #    continue

        #triage_state.update({'randomness': randomness, })

        ## guilty commit
        #message = f"""
        #You are an expert QA, the torch-xpu-ops UT passed on good commit {args.good_commit} but failed on current commit.
        #1. Please run  git log to extract the current commit with command
        #        git log -1 --pretty=format:'%H'
        #   then the commit range is <good_commit>...<current_commit>
        #2. Then run the following commands to get a list of commits in the commit range:\n
        #        git log --pretty=format:'%H' <commit range>
        #3. For each commit in the commit range, check the git show result of that commit with command
        #        git show <commit>
        #   and check the git show output to see whether the commit has any backend specific update on cuda, hpu or xpu, etc.
        #Please return the answer with beutified json format with key "commit" and "evidence", please provide a sentence as evidence.
        #"""

        #output = shell_agent.run(
        #    message,
        #)
        #backend_updated = output_to_dict(output)
        #triage_state.update({'backend_updated': backend_updated, })


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
        

        #Then check the output to see whether the commit updated the the code of function {triage_state["classify"]["original_test_case"]} or updated the implementation of the operation {triage_state["classify"]["torch_op"]}.
        for commit in triage_state['commit_range'].split('\n'):
            print(f"\n\n### check commit {commit}\n")
            message = f"""
            Get the git show result of the commit {commit} with command
                    git show {commit} --function-context

            Then check the output to see whether the commit updated the the code of function {triage_state["classify"]["original_test_case"]} or updated the implementation of the operation 'lu_solve' or 'solve'.
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
