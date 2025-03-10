import sys
import os
import json
import csv
import pandas as pd

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
                        csvData.append([model,acc5])
            # write csv
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(['Model',dt])
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

df_fp32_selected = df_fp32[['Model','fp32']]
df_int8_selected = df_int8[['Model','int8']]
acc_df = pd.merge(df_fp32_selected, df_int8_selected, on='Model') # merge csv files
acc_df['(fp32-int8)/fp32'] = (acc_df['fp32'] - acc_df['int8']) / acc_df['fp32'] # calculation
acc_df['int8/fp32'] = acc_df['int8'] / acc_df['fp32']

acc_df['(fp32-int8)/fp32'] = acc_df['(fp32-int8)/fp32'].apply(lambda x: f"{x:.2%}")  # results percentages

acc_df.to_csv('summary_acc.csv', index=False)  # write to summary_acc.csv

# perf ratio
for filename_perf in os.listdir(work_dir):
    if filename_perf.endswith('.csv') and 'performance' in filename_perf and 'fp32' in filename_perf:
        file_path = os.path.join(work_dir, filename_perf)
        perf_fp32 = pd.read_csv(file_path)
    if filename_perf.endswith('.csv') and 'performance' in filename_perf and 'int8' in filename_perf:
        file_path = os.path.join(work_dir, filename_perf)
        perf_int8 = pd.read_csv(file_path)
# Create Model Data
Model = {
    'Model': ['alexnet','demucs','dlrm','hf_Albert','hf_Bert','hf_Bert_large','hf_DistilBert','hf_Roberta_base','mnasnet1_0',
              'mobilenet_v2','mobilenet_v3_large','nvidia_deeprecommender','pytorch_CycleGAN_and_pix2pix',
              'resnet152','resnet18','resnet50','resnext50_32x4d','shufflenet_v2_x1_0','squeezenet1_1','Super_SloMo',
              'timm_efficientnet','timm_nfnet,timm_regnet','timm_resnest','timm_vision_transformer','timm_vision_transformer_large','timm_vovnet','vgg16']
        }
perf_df = pd.DataFrame(Model)

fp32_merged = pd.merge(perf_df, perf_fp32[['Model', 'Throughput']], on='Model', how='left').rename(columns={'Throughput': 'fp32'})
int8_merged = pd.merge(perf_df, perf_int8[['Model', 'Throughput']], on='Model', how='left').rename(columns={'Throughput': 'int8'})

perf_df = pd.concat([fp32_merged, int8_merged], axis=1)
perf_df = perf_df.loc[:, ~perf_df.columns.duplicated()] #remove extra Model

perf_df['int8/fp32'] = perf_df['int8']/perf_df['fp32']

# write to new csv file
perf_df.to_csv('summary_perf.csv', index=False)

