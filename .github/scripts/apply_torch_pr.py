
import re
import requests
import argparse
import urllib
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--pr-list', '-n', nargs='+',
    default=[
        # Fallback to CPU for XPU FP64
        "https://github.com/pytorch/pytorch/pull/126516",
        # Modify the tolerance level in TIMM benchmark
        "https://github.com/pytorch/pytorch/pull/143739",
        # Fix build error caused by incorrect namespace change by #144014
        "https://github.com/pytorch/pytorch/pull/144450",
    ]
)
parser.add_argument('--extra-pr-list', '-e', nargs='+',default=[])
args = parser.parse_args()


# check reverted PR is in current code base or not
def check_reverted_reopen(pr_info):
    git_cmd = "((git log -n 1 2>&1 |grep 'nightly release' |head -1 |sed 's/.*(//;s/).*//' || true) && git rev-parse HEAD) |head -n 1"
    git_info = subprocess.Popen(git_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    main_commit = git_info.communicate()[0].decode("utf-8").replace("\n", "")
    revert_cmd = "cur_cmt=$(git rev-parse HEAD) && git fetch origin main > /dev/null 2>&1 && " + \
                 "git checkout " + main_commit + " > /dev/null 2>&1 && " + \
                 "git log |grep 'Reverted " + pr_info["html_url"] + " ' || true && " + \
                 "git checkout $cur_cmt > /dev/null 2>&1"
    revert_info = subprocess.Popen(revert_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    revert_msg = revert_info.communicate()[0].decode("utf-8")
    if "Reverted " + pr_info["html_url"] in revert_msg:
        reverted = True
    else:
        reverted = False
    return reverted

def check_merged(pr_info):
    git_cmd = "((git log -n 1 2>&1 |grep 'nightly release' |head -1 |sed 's/.*(//;s/).*//' || true) && git rev-parse HEAD) |head -n 1"
    git_info = subprocess.Popen(git_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    main_commit = git_info.communicate()[0].decode("utf-8").replace("\n", "")
    merge_cmd = "cur_cmt=$(git rev-parse HEAD) && git fetch origin main > /dev/null 2>&1 && " + \
                 "git checkout " + main_commit + " > /dev/null 2>&1 && " + \
                 "git log |grep 'resolved: " + pr_info["html_url"] + "' || true && " + \
                 "git checkout $cur_cmt > /dev/null 2>&1"
    merge_info = subprocess.Popen(merge_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    merge_msg = merge_info.communicate()[0].decode("utf-8")
    if "resolved: " + pr_info["html_url"] in merge_msg:
        merged = True
    else:
        merged = False
    return merged

def appyly_pr(pr_info, re_apply_msg):
    # get pr diff
    pr_file = pr_info["diff_url"].split("/")[-1]
    urllib.request.urlretrieve(pr_info["diff_url"], pr_file)
    # apply diff
    apply_cmd = "git apply --3way " + pr_file
    apply_info = subprocess.Popen(apply_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    apply_message = apply_info.communicate()[0].decode("utf-8")
    apply_status = apply_info.returncode
    # apply status
    if apply_status == 0:
        print("{} {}, applied got SUCCESSFUL".format(pr_info["diff_url"], re_apply_msg))
    else:
        print("{} {}, applied got FAILED".format(pr_info["diff_url"], apply_message))
        print(apply_status, apply_message)
        exit(1)


# headers = {'Authorization': 'Bearer ' + args.token} if args.token != None else args.token
pr_list = args.pr_list + args.extra_pr_list
pr_list = set(pr_list)
pr_list = sorted(pr_list)
for pr_link in pr_list:
    repo_info = pr_link.split("/")
    pr_info = requests.get('https://api.' + repo_info[-5] + '/repos/' + repo_info[-4] + '/' + \
                        repo_info[-3] + '/pulls/' + repo_info[-1], timeout=60).json()

    if pr_info["state"].lower() == "open":
        # for reverted PR
        reverted_id = next((item["id"] for item in pr_info["labels"] if item["name"] == "Reverted"), -1)
        re_apply_msg = "is opened"
        if reverted_id != -1:
            reverted = check_reverted_reopen(pr_info)
            # skip if PR not reverted but re-open in current code base
            if not reverted:
                print("{} is re-open but not reverted, no need to apply".format(pr_info["diff_url"]))
                continue
            else:
                re_apply_msg = "is re-opened and reverted,"
        appyly_pr(pr_info, re_apply_msg)
    elif pr_info["state"].lower() == "closed":
        merged_id = next((item["id"] for item in pr_info["labels"] if item["name"] == "Merged"), -1)
        re_apply_msg = "is closed but not merged"
        if merged_id != -1:
            merged = check_merged(pr_info)
            if merged:
                print("{} is closed and merged, no need to apply".format(pr_info["diff_url"]))
                continue
        appyly_pr(pr_info, re_apply_msg)
    else:
        print("{} is {}, no need to apply".format(pr_info["diff_url"], pr_info["state"]))
        exit(1)

