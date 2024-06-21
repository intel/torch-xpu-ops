import argparse
import pandas as pd
from scipy.stats import gmean
from styleframe import StyleFrame, Styler, utils

parser = argparse.ArgumentParser(description="Generate report")
parser.add_argument('-s', '--suite', default='huggingface', choices=["torchbench", "huggingface", "timm_models"], type=str, help='model suite name')
parser.add_argument('-p', '--precision', default=["amp_fp16", "float32"], nargs='*', type=str, help='precision')
#parser.add_argument('-t', '--target', type=str, help='target log files')
parser.add_argument('-r', '--reference', type=str, help='reference log files')
parser.add_argument('-m', '--mode', default=["inference", "training"], nargs='*', type=str, help='mode name')
args = parser.parse_args()

passrate_values = {}

failure_style = Styler(bg_color='#FF0000', font_color=utils.colors.black)
regression_style = Styler(bg_color='#F0E68C', font_color=utils.colors.red)
improve_style = Styler(bg_color='#00FF00', font_color=utils.colors.black)


def percentage(part, whole, decimals=2):
    if whole == 0:
        return 0
    return round(100 * float(part) / float(whole), decimals)

def get_passing_entries(df, column_name):
    return df[df[column_name].notnull()]

def caculate_passrate(df, key_word):
    # some models may be new for reference
    df = get_passing_entries(df, key_word)
    total = len(df.index)
    passing = df[df[key_word].fillna('').str.contains('pass')][key_word].count()
    perc = int(percentage(passing, total, decimals=0))
    return f"{perc}%, {passing}/{total}"

# def validate_csv_files(csv_file):
#     df = pd.read_csv(csv_file,header=None)
#     expected_header = ['dev', 'name', 'batch_size', 'accuracy', 'calls_captured','unique_graphs','graph_breaks','unique_graph_breaks']
#     if df.iloc[0].tolist() == expected_header:
#         print(f"{csv_file} Expected csv file")
#         return True
#     else:
#         print(f"{csv_file} file no right header!!!")
#         return False

def get_acc_csv(precision, mode):
    target_path = 'inductor_log/' + args.suite + '/' + precision + '/inductor_' + args.suite + '_' + precision + '_' + mode + '_xpu_accuracy.csv'
    #target_data = pd.DataFrame()
    #if validate_csv_files(target_path):
    target_ori_data = pd.read_csv(target_path)
    target_data = target_ori_data.copy()
    target_data.sort_values(by=['name'])
    # else:
    #    print("file skip")
    
    if args.reference is not None:
        reference_file_path = args.reference + '/inductor_log/' + args.suite + '/' + precision + '/inductor_' + args.suite + '_' + precision + '_' + mode + '_xpu_accuracy.csv'
        # if validate_csv_files(reference_file_path):
        reference_ori_data = pd.read_csv(reference_file_path)
        reference_data = reference_ori_data.copy()
        reference_data.sort_values(by=['name'])
        data = pd.merge(target_data,reference_data,on=['name'],how= 'outer')
        return data
        # else:
        #     print("file skip")
    else:
        return target_data

def process(input, precision, mode):
    global passrate_values
    if input is not None:
        if args.reference is None:
            data_new = input[['name', 'batch_size', 'accuracy']].rename(columns={'name': 'name', 'batch_size': 'batch_size', 'accuracy': 'accuracy'})
            passrate_values['target_' + str(precision) + '_' + str(mode)] = caculate_passrate(data_new, 'accuracy')
            data = StyleFrame({'name': list(data_new['name']),
                            'batch_size': list(data_new['batch_size']),
                            'accuracy': list(data_new['accuracy'])})
            data.set_column_width(1, 10)
            data.set_column_width(2, 18)
            data.set_column_width(3, 18)
            data.set_row_height(rows=data.row_indexes, height=15)
        else:
            data_new=input[['name','batch_size_x','accuracy_x']].rename(columns={'name':'name','batch_size_x':'batch_size_new','accuracy_x':'accuracy_new'})
            passrate_values['target_' + str(precision) + '_' + str(mode)] = caculate_passrate(data_new, 'accuracy_new')        
            data_old=input[['batch_size_y','accuracy_y']].rename(columns={'batch_size_y':'batch_size_old','accuracy_y':'accuracy_old'}) 
            passrate_values['reference_' + str(precision) + '_' + str(mode)] = caculate_passrate(data_old, 'accuracy_old')
            data_comp = pd.DataFrame((data_new['accuracy_new'] != 'pass') & (data_old['accuracy_old'] == 'pass'),columns=['Accuracy regression'])
            combined_data = pd.DataFrame({
                'name': list(data_new['name']),
                'batch_size_new': list(data_new['batch_size_new']),
                'accuracy_new': list(data_new['accuracy_new']),
                'batch_size_old': list(data_old['batch_size_old']),
                'accuracy_old': list(data_old['accuracy_old']),
                'Accuracy regression': list(data_comp['Accuracy regression'])
                })   
            data = StyleFrame(combined_data)
            data.set_column_width(1, 10)
            data.set_column_width(2, 18) 
            data.set_column_width(3, 18) 
            data.set_column_width(4, 18)
            data.set_column_width(5, 15)
            data.set_column_width(6, 20)
            data.apply_style_by_indexes(indexes_to_style=data[(data['Accuracy regression'] == 'regression')],styler_obj=regression_style)
            data.set_row_height(rows=data.row_indexes, height=15)        
        return data
    else:
        return pd.DataFrame()

def update_details(precision, mode, excel):
    h = {"A": 'Model suite', "B": '', "C": "target", "D": '', "E": args.reference, "F": '', "G": 'Result Comp'}
    if args.reference is None:
        h = {"A": 'Model suite', "B": '', "C": "target", "D": ''}
    head = StyleFrame(pd.DataFrame(h, index=[0]))
    head.set_column_width(1, 15)
    head.set_row_height(rows=[1], height=15)

    head.to_excel(excel_writer=excel, sheet_name=precision + '_' + mode, index=False, startrow=0, header=False)
    target_raw_data = get_acc_csv(precision, mode)
    target_data = process(target_raw_data, precision, mode)
    target_data.to_excel(excel_writer=excel, sheet_name=precision + '_' + mode, index=False, startrow=1, startcol=1)

def update_summary(excel):
    data = {
        'Test Secnario': ['AMP_BF16 Inference', 'AMP_BF16 Training', 'AMP_FP16 Inference', 'AMP_FP16 Training', 'BF16 Inference', 'BF16 Training', 'FP16 Inference', 'FP16 Training', 'FP32 Inference', 'FP32 Training'],
        'Comp Item': ['Pass Rate',  'Pass Rate',  'Pass Rate',  'Pass Rate',  'Pass Rate',  'Pass Rate',  'Pass Rate',  'Pass Rate',  'Pass Rate',  'Pass Rate'],
        'compiler': ['inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor', 'inductor'],
        'torchbench': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        'huggingface': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        'timm_models ': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        'refer_torchbench ': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        'refer_huggingface ': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        'refer_timm_models ': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
    }
    summary = pd.DataFrame(data)

    if 'huggingface' in args.suite:
        if 'amp_bf16' in args.precision:
            if 'inference' in args.mode:
                # passrate
                summary.iloc[0:1, 4:5] = passrate_values['target_amp_bf16_inference']
            if 'training' in args.mode:
                summary.iloc[1:2, 4:5] = passrate_values['target_amp_bf16_training']
                
        if 'amp_fp16' in args.precision:
            if 'inference' in args.mode:
                summary.iloc[2:3, 4:5] = passrate_values['target_amp_fp16_inference']
            if 'training' in args.mode:
                summary.iloc[3:4, 4:5] = passrate_values['target_amp_fp16_training']

        if 'bfloat16' in args.precision:
            if 'inference' in args.mode:
                summary.iloc[4:5, 4:5] = passrate_values['target_bfloat16_inference']
            if 'training' in args.mode:
                summary.iloc[5:6, 4:5] = passrate_values['target_bfloat16_training']
        if 'float16' in args.precision:
            if 'inference' in args.mode:
                summary.iloc[6:7, 4:5] = passrate_values['target_float16_inference']
            if 'training' in args.mode:
                summary.iloc[7:8, 4:5] = passrate_values['target_float16_training']
        if 'float32' in args.precision:
            if 'inference' in args.mode:
                summary.iloc[8:9, 4:5] = passrate_values['target_float32_inference']
            if 'training' in args.mode:
                summary.iloc[9:10, 4:5] = passrate_values['target_float32_training']

        if args.reference is not None:
            if 'amp_bf16' in args.precision:
                if 'inference' in args.mode:
                # passrate
                    summary.iloc[0:1, 7:8] = passrate_values['reference_amp_bf16_inference']
                if 'training' in args.mode:
                    summary.iloc[1:2, 7:8] = passrate_values['reference_amp_bf16_training']
            if 'amp_fp16' in args.precision:
                if 'inference' in args.mode:
                    summary.iloc[2:3, 7:8] = passrate_values['reference_amp_fp16_inference']
                if 'training' in args.mode:
                    summary.iloc[3:4, 7:8] = passrate_values['reference_amp_fp16_training']

            if 'bfloat16' in args.precision:
                if 'inference' in args.mode:
                    summary.iloc[4:5, 7:8] = passrate_values['reference_bfloat16_inference']
                if 'training' in args.mode:
                    summary.iloc[5:6, 7:8] = passrate_values['reference_bfloat16_training']
            if 'float16' in args.precision:
                if 'inference' in args.mode:
                    summary.iloc[6:7, 7:8] = passrate_values['reference_float16_inference']
                if 'training' in args.mode:
                    summary.iloc[7:8, 7:8] = passrate_values['reference_float16_training']
            if 'float32' in args.precision:
                if 'inference' in args.mode:
                    summary.iloc[8:9, 7:8] = passrate_values['reference_float32_inference']
                if 'training' in args.mode:
                    summary.iloc[9:10, 7:8] = passrate_values['reference_float32_training']
        
        print("====================Summary=============================")
        print(summary)
        
        sf = StyleFrame(summary)
        for i in range(1, 10):
            sf.set_column_width(i, 22)
        for j in range(1, 12):
            sf.set_row_height(j, 30)
        sf.to_excel(sheet_name='Summary', excel_writer=excel)

    if 'timm_models' in args.suite:
        if 'amp_bf16' in args.precision:
            if 'inference' in args.mode:
                # passrate
                summary.iloc[0:1, 5:6] = passrate_values['target_amp_bf16_inference']
            if 'training' in args.mode:
                summary.iloc[1:2, 5:6] = passrate_values['target_amp_bf16_training']
        if 'amp_fp16' in args.precision:
            if 'inference' in args.mode:
                summary.iloc[2:3, 5:6] = passrate_values['target_amp_fp16_inference']
            if 'training' in args.mode:
                summary.iloc[3:4, 5:6] = passrate_values['target_amp_fp16_training']

        if 'bfloat16' in args.precision:
            if 'inference' in args.mode:
                summary.iloc[4:5, 5:6] = passrate_values['target_bfloat16_inference']
            if 'training' in args.mode:
                summary.iloc[5:6, 5:6] = passrate_values['target_bfloat16_training']
        if 'float16' in args.precision:
            if 'inference' in args.mode:
                summary.iloc[6:7, 5:6] = passrate_values['target_float16_inference']
            if 'training' in args.mode:
                summary.iloc[7:8, 5:6] = passrate_values['target_float16_training']
        if 'float32' in args.precision:
            if 'inference' in args.mode:
                summary.iloc[8:9, 5:6] = passrate_values['target_float32_inference']
            if 'training' in args.mode:
                summary.iloc[9:10, 5:6] = passrate_values['target_float32_training']

        if args.reference is not None:
            if 'amp_bf16' in args.precision:
                if 'inference' in args.mode:
                    # passrate
                    summary.iloc[0:1, 8:9] = passrate_values['reference_amp_bf16_inference']
                if 'training' in args.mode:
                    summary.iloc[1:2, 8:9] = passrate_values['reference_amp_bf16_training']
            if 'amp_fp16' in args.precision:
                if 'inference' in args.mode:
                    summary.iloc[2:3, 8:9] = passrate_values['reference_amp_fp16_inference']
                if 'training' in args.mode:
                    summary.iloc[3:4, 8:9] = passrate_values['reference_amp_fp16_training']

            if 'bfloat16' in args.precision:
                if 'inference' in args.mode:
                    summary.iloc[4:5, 8:9] = passrate_values['reference_bfloat16_inference']
                if 'training' in args.mode:
                    summary.iloc[5:6, 8:9] = passrate_values['reference_bfloat16_training']
            if 'float16' in args.precision:
                if 'inference' in args.mode:
                    summary.iloc[6:7, 8:9] = passrate_values['reference_float16_inference']
                if 'training' in args.mode:
                    summary.iloc[7:8, 8:9] = passrate_values['reference_float16_training']
            if 'float32' in args.precision:
                if 'inference' in args.mode:
                    summary.iloc[8:9, 8:9] = passrate_values['reference_float32_inference']
                if 'training' in args.mode:
                    summary.iloc[9:10, 8:9] = passrate_values['reference_float32_training']
        
        print("====================Summary=============================")
        print(summary)
        
        sf = StyleFrame(summary)
        for i in range(1, 10):
            sf.set_column_width(i, 22)
        for j in range(1, 12):
            sf.set_row_height(j, 30)
        sf.to_excel(sheet_name='Summary', excel_writer=excel)
    
    if 'torchbench' in args.suite:
        if 'amp_bf16' in args.precision:
            if 'inference' in args.mode:
                # passrate
                summary.iloc[0:1, 3:4] = passrate_values['target_amp_bf16_inference']
            if 'training' in args.mode:
                summary.iloc[1:2, 3:4] = passrate_values['target_amp_bf16_training']
        if 'amp_fp16' in args.precision:
            if 'inference' in args.mode:
                summary.iloc[2:3, 3:4] = passrate_values['target_amp_fp16_inference']
            if 'training' in args.mode:
                summary.iloc[3:4, 3:4] = passrate_values['target_amp_fp16_training']

        if 'bfloat16' in args.precision:
            if 'inference' in args.mode:
                summary.iloc[4:5, 3:4] = passrate_values['target_bfloat16_inference']
            if 'training' in args.mode:
                summary.iloc[5:6, 3:4] = passrate_values['target_bfloat16_training']
        if 'float16' in args.precision:
            if 'inference' in args.mode:
                summary.iloc[6:7, 3:4] = passrate_values['target_float16_inference']
            if 'training' in args.mode:
                summary.iloc[7:8, 3:4] = passrate_values['target_float16_training']
        if 'float32' in args.precision:
            if 'inference' in args.mode:
                summary.iloc[8:9, 3:4] = passrate_values['target_float32_inference']
            if 'training' in args.mode:
                summary.iloc[9:10, 3:4] = passrate_values['target_float32_training']

        if args.reference is not None:
            if 'amp_bf16' in args.precision:
                if 'inference' in args.mode:
                    # passrate
                    summary.iloc[0:1, 6:7] = passrate_values['reference_amp_bf16_inference']
                if 'training' in args.mode:
                    summary.iloc[1:2, 6:7] = passrate_values['reference_amp_bf16_training']

            if 'amp_fp16' in args.precision:
                if 'inference' in args.mode:
                    summary.iloc[2:3, 6:7] = passrate_values['reference_amp_fp16_inference']
                if 'training' in args.mode:
                    summary.iloc[3:4, 6:7] = passrate_values['reference_amp_fp16_training']

            if 'bfloat16' in args.precision:
                if 'inference' in args.mode:
                    summary.iloc[4:5, 6:7] = passrate_values['reference_bfloat16_inference']
                if 'training' in args.mode:
                    summary.iloc[5:6, 6:7] = passrate_values['reference_bfloat16_training']
            if 'float16' in args.precision:
                if 'inference' in args.mode:
                    summary.iloc[6:7, 6:7] = passrate_values['reference_float16_inference']
                if 'training' in args.mode:
                    summary.iloc[7:8, 6:7] = passrate_values['reference_float16_training']
            if 'float32' in args.precision:
                if 'inference' in args.mode:
                    summary.iloc[8:9, 6:7] = passrate_values['reference_float32_inference']
                if 'training' in args.mode:
                    summary.iloc[9:10, 6:7] = passrate_values['reference_float32_training']
        
        print("====================Summary=============================")
        print(summary)
        
        sf = StyleFrame(summary)
        for i in range(1, 10):
            sf.set_column_width(i, 22)
        for j in range(1, 12):
            sf.set_row_height(j, 30)
        sf.to_excel(sheet_name='Summary', excel_writer=excel)

def generate_report(excel, precision_list, mode_list):
    for p in precision_list:
        for m in mode_list:
            update_details(p, m, excel)
    update_summary(excel)


def excel_postprocess(file, precison, mode):
    wb = file.book
    # details
    for p in precison:
        for m in mode:
            wdt = wb[p + '_' + m]
            wdt.merge_cells(start_row=1, end_row=2, start_column=1, end_column=1)
            wdt.merge_cells(start_row=1, end_row=1, start_column=3, end_column=4)
            wdt.merge_cells(start_row=1, end_row=1, start_column=5, end_column=6)
            wdt.merge_cells(start_row=1, end_row=1, start_column=7, end_column=7)
    wb.save(file)


if __name__ == '__main__':
    summary_path = 'inductor_log/' + str(args.suite) + '/Inductor_' + args.suite + '_E2E_Test_Acc_Report.xlsx'
    excel = StyleFrame.ExcelWriter(summary_path)
    print(f"=========Acc Check Summary file located in {summary_path}==================")
    generate_report(excel, args.precision, args.mode)
    excel_postprocess(excel, args.precision, args.mode)
