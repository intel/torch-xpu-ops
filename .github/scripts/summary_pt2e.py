import sys
import os
import json
import csv
import pandas as pd


work_dir = sys.argv[1]
# scan files endwith .log and accuracy in folder
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
            with open(log_file_path, 'r', encoding='utf-8') as log_file:
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
                    with open(json_file_path, 'r', encoding='utf-8') as json_file:
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
                
