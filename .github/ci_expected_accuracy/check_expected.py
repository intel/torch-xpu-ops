
import argparse
import pandas as pd
import pathlib


parser = argparse.ArgumentParser(description="Accuracy Check", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--category", type=str, default="inductor", help="inductor")
parser.add_argument("--suite", type=str, required=True, help="huggingface, timm_models or torchbench")
parser.add_argument("--mode", type=str, required=True, help="inference or training")
parser.add_argument("--dtype", type=str, required=True, help="float32, bfloat16, float16, amp_bf16 or amp_fp16")
# parser.add_argument("--scenario", type=str, required=True, help="accuracy or performance")
parser.add_argument("--csv_file", type=str, required=True, help="your result csv path")
parser.add_argument('--update', action='store_true', help="whether update new pass and new failed info")
args = parser.parse_args()


# load csv files
test_data= pd.read_csv(args.csv_file)
# test_data = test_data.reset_index()  # make sure indexes pair with number of rows
# test_data = test_data.sort_values(by=["name"], ascending=True)
test_names = [row["name"] for index, row in test_data.iterrows()]

current_path = pathlib.Path(__file__).parent.resolve()
refer_file = str(current_path) + "/" + args.category + "_" + args.suite + "_" + args.mode + ".csv"
refer_data= pd.read_csv(refer_file)
# refer_data = refer_data.reset_index()  # make sure indexes pair with number of rows
# refer_data = refer_data.sort_values(by=["name"], ascending=True)
refer_names = [row["name"] for index, row in refer_data.iterrows()]

# summary
new_pass_models = []
failed_models = []
for index, row in refer_data.iterrows():
    test_accuracy = next((line["accuracy"] for i, line in test_data.iterrows() if line["name"] == row["name"]), "None")
    if test_accuracy == "None":
        failed_models.append([row["name"], "N/A"])
    elif test_accuracy != row[args.dtype] and "pass" not in test_accuracy and "pass" not in row[args.dtype]:
        failed_models.append([row["name"], test_accuracy])
        # update failed info for reference data
        refer_data.at[index, args.dtype] = test_accuracy
    elif test_accuracy != row[args.dtype] and "pass" not in test_accuracy:
        failed_models.append([row["name"], test_accuracy])
    elif test_accuracy != row[args.dtype] and "pass" in test_accuracy:
        new_pass_models.append([row["name"], test_accuracy])
        # update new pass for reference data
        refer_data.at[index, args.dtype] = test_accuracy

# pass rate
total = len(refer_names)
failed = len(failed_models)
pass_rate = 1 - (failed / total)
print("============ Summary for {} {} {} accuracy ============".format(args.suite, args.dtype, args.mode))
print("Total:", total)
print("Failed:", failed, failed_models)
print("Passed", total - failed)
print("Pass rate: {:.2f}%".format(pass_rate * 100))

if len(new_pass_models) > 0:
    print("NOTE: New passed models, please update the reference", new_pass_models)
    if args.update:
        refer_data.to_csv(refer_file, sep=',', encoding='utf-8', index=False)
        print("Reference data updated")
