
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
test_data= pd.read_csv(args.csv_file, comment='#')
# test_data = test_data.reset_index()  # make sure indexes pair with number of rows
# test_data = test_data.sort_values(by=["name"], ascending=True)
test_names = [row["name"] for index, row in test_data.iterrows()]

current_path = pathlib.Path(__file__).parent.resolve()
refer_file = str(current_path) + "/" + args.category + "_" + args.suite + "_" + args.mode + ".csv"
refer_data= pd.read_csv(refer_file, comment='#')
# refer_data = refer_data.reset_index()  # make sure indexes pair with number of rows
# refer_data = refer_data.sort_values(by=["name"], ascending=True)
refer_names = [row["name"] for index, row in refer_data.iterrows()]

# summary
model_names = set(refer_names + test_names)
passed_models = []
real_failed_models = []
expected_failed_models = []
new_models = []
new_pass_models = []
lost_models = []
timeout_models = []
for model_name in model_names:
# for index, row in refer_data.iterrows():
    test_row = next(([i, line] for i, line in test_data.iterrows() if line["name"] == model_name), "N/A")
    refer_row = next(([i, line] for i, line in refer_data.iterrows() if line["name"] == model_name), "N/A")
    test_accuracy = test_row[1]["accuracy"] if test_row != "N/A" else "N/A"
    refer_accuracy = refer_row[1][args.dtype] if refer_row != "N/A" else "N/A"
    test_accuracy = str(test_accuracy)
    refer_accuracy = str(refer_accuracy)
    if test_accuracy == "N/A":
        lost_models.append([model_name, test_accuracy])
    elif 'pass' in test_accuracy:
        passed_models.append([model_name, test_accuracy])
        if refer_accuracy == "N/A":
            new_models.append([model_name, test_accuracy])
            refer_data.loc[len(refer_data),:] = "N/A"
            refer_data.at[len(refer_data) - 1, "name"] = model_name
            refer_data.at[len(refer_data) - 1, args.dtype] = test_accuracy
        elif 'pass' not in refer_accuracy:
            new_pass_models.append([model_name, test_accuracy])
            refer_data.at[refer_row[0], args.dtype] = test_accuracy
    elif 'timeout' in test_accuracy:
        timeout_models.append([model_name, test_accuracy])
        if refer_accuracy == "N/A":
            new_models.append([model_name, test_accuracy])
            refer_data.loc[len(refer_data),:] = "N/A"
            refer_data.at[len(refer_data) - 1, "name"] = model_name
            refer_data.at[len(refer_data) - 1, args.dtype] = test_accuracy
    else:
        if refer_accuracy == "N/A":
            new_models.append([model_name, test_accuracy])
            real_failed_models.append([model_name, test_accuracy])
            refer_data.loc[len(refer_data),:] = "N/A"
            refer_data.at[len(refer_data) - 1, "name"] = model_name
            refer_data.at[len(refer_data) - 1, args.dtype] = test_accuracy
        elif "pass" in refer_accuracy:
            real_failed_models.append([model_name, test_accuracy])
        else:
            expected_failed_models.append([model_name, test_accuracy])
            if test_accuracy != refer_accuracy:
                refer_data.at[refer_row[0], args.dtype] = test_accuracy

# pass rate
print("============ Summary for {} {} {} accuracy ============".format(args.suite, args.dtype, args.mode))
print("Total models:", len(model_names))
print("Passed models:", len(passed_models))
print("Real failed models:", len(real_failed_models), real_failed_models)
print("Expected failed models:", len(expected_failed_models), expected_failed_models)
print("Warning timeout models:", len(timeout_models), timeout_models)
print("New models:", len(new_models), new_models)
print("Failed to passed models:", len(new_pass_models), new_pass_models)
print("Not run/in models:", len(lost_models), lost_models)
print("Pass rate: {:.2f}%".format(len(passed_models) / len(model_names) * 100))

if len(new_pass_models + new_models) > 0:
    print("NOTE: New models result, please update the reference", new_pass_models, new_models)
    if args.update:
        refer_data.to_csv(refer_file, sep=',', encoding='utf-8', index=False)
        print("Updated. Now, confirm the changes to .csvs and `git add` them if satisfied.")
