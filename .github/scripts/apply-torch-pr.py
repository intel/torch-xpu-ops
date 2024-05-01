
import re
import requests
import argparse
import urllib
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--pr-list', '-n', nargs='+', default=[], required=True)
parser.add_argument('--repo', '-r', default='https://github.com/pytorch/pytorch.git', type=str)
args = parser.parse_args()


repo_info = re.sub(r'\.git$', '', args.repo.strip()).split("/")
# headers = {'Authorization': 'Bearer ' + args.token} if args.token != None else args.token
for pr_number in args.pr_list:
    pr_info = requests.get('https://api.' + repo_info[-3] + '/repos/' + repo_info[-2] + '/' + \
                        repo_info[-1] + '/pulls/' + pr_number, timeout=60).json()

    if pr_info["state"].lower() == "open":
        pr_file = pr_info["diff_url"].split("/")[-1]
        urllib.request.urlretrieve(pr_info["diff_url"], pr_file)
        apply_cmd = "git apply --3way " + pr_file + " && rm -f " + pr_file
        print(apply_cmd)
        apply_info = subprocess.Popen(apply_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        apply_message = apply_info.communicate()[0].decode("utf-8")
        apply_status = apply_info.returncode
        if apply_status == 0:
            print("{} applied got SUCCESSFUL".format(pr_info["diff_url"]))
        else:
            print("{} applied got FAILED".format(pr_info["diff_url"]))
            print(apply_status, apply_message)
            exit(1)
    elif pr_info["state"].lower() == "closed":
        print("{} is ClOSED, no need to apply".format(pr_info["diff_url"]))
    else:
        print("{} is UNKNOWN, no need to apply".format(pr_info["diff_url"]))
        exit(1)

