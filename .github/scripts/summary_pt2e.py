import sys
import os
import json
import csv
import pandas as pd

def get_model_result():
    if os.path.exists('summary_acc.csv'):
        print("###Accuracy")
        print("| Model | fp32 Acc@1 | fp32 Acc@5 | int8 Acc@1 | int8 Acc@5 | int8/fp32 Acc@1 | int8/fp32 Acc@5 |")
        data_acc = pd.read_csv('summary_acc.csv')
        data_acc['int8/fp32   Acc@1'] = data_acc['int8/fp32   Acc@1'].apply(lambda x: f"{x * 100:.2f}%")
        data_acc['int8/fp32   Acc@5'] = data_acc['int8/fp32   Acc@5'].apply(lambda x: f"{x * 100:.2f}%")
        data_acc.iloc[:, 1:] = data_acc.iloc[:, 1:].round(2)
        for index, row in data_acc.iterrows():
            print(f"| {row['Model']} | {row['fp32   Acc@1']} | {row['fp32   Acc@5']} | {row['int8   Acc@1']} | {row['int8   Acc@5']} | {row['int8/fp32   Acc@1']} | {row['int8/fp32   Acc@5']} |")
    if os.path.exists('summary_perf.csv'):
        print("###Performance")
        print("| Model | fp32 | int8_ASYMM | int8_SYMM | ASYMM  int8/fp32 | SYMM  int8/fp32 |")
        data_perf = pd.read_csv('summary_perf.csv')
        data_perf.iloc[:, 1:] = data_perf.iloc[:, 1:].round(2)
        for index, row in data_perf.iterrows():
            row_data = [row['Model']] + [f"{x:.2f}" if pd.notna(x) else "NULL" for x in row[1:]]
            print(f"| {row_data[0]} | {row_data[1]} | {row_data[2]} | {row_data[3]} | {row_data[4]} | {row_data[5]} | ")
    # Acc
    html_table_acc = '<table Accuracy>\n'
    html_table_acc += '  <thead>\n'
    html_table_acc += '    <tr>\n'
    headers = data_acc.columns
    for header in headers:
        html_table_acc += f'    <th>{header}</th>\n'
    html_table_acc += '  </tr>\n'
    html_table_acc += '  </thead>\n'
    html_table_acc += '  <tbody>\n'
    for index, row in data_acc.iterrows():
        html_table_acc += '  <tr>\n'
        for value in row:
            html_table_acc += f'    <td>{value}</td>\n'
        html_table_acc += '  </tr>\n'
    html_table_acc += '  </tbody>\n'
    html_table_acc += '</table>'
    # Perf
    html_table_perf = '<table Performance>\n'
    html_table_perf += '  <thead>\n'
    html_table_perf += '    <tr>\n'
    headers = data_perf.columns
    for header in headers:
        html_table_perf += f'    <th>{header}</th>\n'
    html_table_perf += '  </tr>\n'
    html_table_perf += '  </thead>\n'
    html_table_perf += '  <tbody>\n'
    for index, row in data_perf.iterrows():
        html_table_perf += '  <tr>\n'
        row_data = [row['Model']] + [f"{x:.2f}" if pd.notna(x) else "NULL" for x in row[1:]]
        for value in row_data:
            html_table_perf += f'    <td>{value}</td>\n'
        html_table_perf += '  </tr>\n'
    html_table_perf += '  </tbody>\n'
    html_table_perf += '</table>'

    summary_file = os.getenv('GITHUB_STEP_SUMMARY')
    if summary_file:
        with open(summary_file, 'a') as f:
            f.write("PT2E Accuracy Result\n")
            f.write(html_table_acc)
            f.write("PT2E Performance Result\n")
            f.write(html_table_perf)

def main():
    work_dir = sys.argv[1]
    # scan files endwith .log and accuracy in file
    for root, dirs, files in os.walk(work_dir):
        for file in files:
            if file.endswith('.log') and 'accuracy' in file:
                log_file_path = os.path.join(root, file)
                # generate related csv file
                csv_file_name = os.path.splitext(file)[0] + '.csv'
                csv_file_path = os.path.join(root, csv_file_name)
                # Data
                csvData = []
                # read log
                with open(log_file_path, encoding='utf-8') as log_file:
                    for line in log_file:
                        if "Acc" in line:
                            parts = line.strip().split()
                            model = parts[0].rstrip(':')
                            dt = parts[1].rstrip(':')
                            acc1 = parts[4]
                            acc5 = parts[6]
                            csvData.append([model,acc1,acc5])
                # write csv
                    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(['Model', dt + '   Acc@1', dt + '   Acc@5'])
                        writer.writerows(csvData)

    # scan .json file
    for item in os.listdir(work_dir):
        item_path = os.path.join(work_dir, item)
        if os.path.isdir(item_path):
            # generate csv
            csv_file_name = item + '.csv'
            csv_file_path = os.path.join(work_dir, csv_file_name)
            # data
            csvData = []
            # scan json
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    if file.endswith('.json'):
                        json_file_path = os.path.join(root, file)
                        with open(json_file_path, encoding='utf-8') as json_file:
                            data = json.load(json_file)
                            metrics = data.get('metrics',{})
                            try:
                                for key, value in metrics.items():
                                    parts = key.rsplit('-eval_throughput',1)
                                    if len(parts) == 2:
                                        model = parts[0]
                                        throughput = value
                                        csvData.append([model,throughput])
                            except json.JSONDecodeError:
                                print(f"Error decoding JSON file: {json_file_path}")

                with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['Model','Throughput'])
                    writer.writerows(csvData)
    # accuracy ratio
    for filename in os.listdir(work_dir):
        if filename.endswith('.csv') and 'accuracy' in filename and 'fp32' in filename:
            file_path = os.path.join(work_dir, filename)
            df_fp32 = pd.read_csv(file_path)
        if filename.endswith('.csv') and 'accuracy' in filename and 'int8' in filename:
            file_path = os.path.join(work_dir, filename)
            df_int8 = pd.read_csv(file_path)

    df_fp32_selected = df_fp32[['Model','fp32   Acc@1', 'fp32   Acc@5']]
    df_int8_selected = df_int8[['Model','int8   Acc@1', 'int8   Acc@5']]
    acc_df = pd.merge(df_fp32_selected, df_int8_selected, on='Model') # merge csv files
    acc_df['int8/fp32   Acc@1'] = acc_df['int8   Acc@1'] / acc_df['fp32   Acc@1']
    acc_df['int8/fp32   Acc@5'] = acc_df['int8   Acc@5'] / acc_df['fp32   Acc@5']

    acc_df.to_csv('summary_acc.csv', index=False)  # write to summary_acc.csv

    # perf ratio
    for filename_perf in os.listdir(work_dir):
        if filename_perf.endswith('.csv') and 'performance' in filename_perf and 'fp32' in filename_perf:
            file_path = os.path.join(work_dir, filename_perf)
            perf_fp32 = pd.read_csv(file_path)
        if filename_perf.endswith('.csv') and 'performance' in filename_perf and 'int8' and 'ASYMM' in filename_perf:
            file_path = os.path.join(work_dir, filename_perf)
            perf_int8_asymm = pd.read_csv(file_path)
        if filename_perf.endswith('.csv') and 'performance' in filename_perf and 'int8' and 'SYMM' in filename_perf:
            file_path = os.path.join(work_dir, filename_perf)
            perf_int8_symm = pd.read_csv(file_path)
    # Create Model Data
    Model = {
        'Model': ['alexnet','demucs','dlrm','hf_Albert','hf_Bert','hf_Bert_large','hf_DistilBert','hf_Roberta_base','mnasnet1_0',
                  'mobilenet_v2','mobilenet_v3_large','nvidia_deeprecommender','pytorch_CycleGAN_and_pix2pix',
                  'resnet152','resnet18','resnet50','resnext50_32x4d','shufflenet_v2_x1_0','squeezenet1_1','Super_SloMo',
                  'timm_efficientnet','timm_nfnet','timm_regnet','timm_resnest','timm_vision_transformer','timm_vision_transformer_large','timm_vovnet','vgg16']
            }
    perf_df = pd.DataFrame(Model)

    fp32_merged = pd.merge(perf_df, perf_fp32[['Model', 'Throughput']], on='Model', how='left').rename(columns={'Throughput': 'fp32'})
    int8_asymm_merged = pd.merge(perf_df, perf_int8_asymm[['Model', 'Throughput']], on='Model', how='left').rename(columns={'Throughput': 'int8_ASYMM'})
    int8_symm_merged = pd.merge(perf_df, perf_int8_symm[['Model', 'Throughput']], on='Model', how='left').rename(columns={'Throughput': 'int8_SYMM'})

    perf_df = pd.concat([fp32_merged, int8_asymm_merged,int8_symm_merged], axis=1)
    perf_df = perf_df.loc[:, ~perf_df.columns.duplicated()] #remove extra Model

    perf_df['ASYMM  int8/fp32'] = perf_df['int8_ASYMM']/perf_df['fp32']
    perf_df['SYMM  int8/fp32'] = perf_df['int8_SYMM']/perf_df['fp32']

    # write to new csv file
    perf_df.to_csv('summary_perf.csv', index=False)
    get_model_result()

if __name__ == "__main__":
    main()