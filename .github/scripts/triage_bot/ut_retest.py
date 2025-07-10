import os
import json
import ast
from trace_parser import TraceParser
from visit_ast import VisitAST
import shutil

fail_list_path = "ut_failure_list.txt"

def copy_file(source, dest):
    """
    Copy file from source to dest.

    Args:
        source: The source file to copy
        dest: The destination file
    """
    try:
        shutil.copyfile(source, dest)
    except FileNotFoundError:
        print(f"Error: Source file '{source}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def instrument(fail_list_path):
    """
    Get the stack of each failure in the fail_list_path, for each failure,
    instrument the files on trackback stack with debug information.
    The  original file is backuped to <file>.bk and will be updated with an instrumented version.

    Args:
        fail_list_path: the fail list file with json format
    """
    with open(fail_list_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
            except Exception as e:
                print(f"An error occurred: {e}")

            # Parse traceback
            traceback = data['Traceback']
            trace_parser = TraceParser(traceback)
            trace_parser.parse_log_traceback()

            # Instrument the files on stack with debug information
            stack_dict = {}
            for stack in trace_parser.trace_dict['stack']:
                file = stack['file']
                line = stack['line']
                func = stack['function']
                if file not in stack_dict.keys():
                    stack_dict[file] = {line: func,}
                else:
                    _dict = stack_dict[file]
                    _dict |= {line: func,}
                    stack_dict[file].update(_dict)

            for file in stack_dict.keys():
                file_bk = file + '.bk'
                file_retest = file + 'retest'

                if os.path.exists(file_bk):
                    copy_file(file_bk, file)

                with open(file, 'r') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    transformer = VisitAST(stack_dict[file])
                    new_tree = transformer.visit(tree)
                    ast.fix_missing_locations(new_tree)

                    copy_file(file, file_bk)

                    with open(file_retest, 'w') as f_retest:
                        f_retest.write(ast.unparse(new_tree))

                    copy_file(file_retest, file)
                    print(f"file {file} is instrumented.") 


instrument(fail_list_path)


