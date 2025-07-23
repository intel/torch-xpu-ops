from skip_list_common import skip_dict
file_b_path = "cuda.csv"
file_c_path = "torch-xpu-ops.csv"
lines_a = []
lines_a_f = []
with open('status.csv', 'w') as f:
    with open("xpu.csv") as fa:
        lines = fa.readlines()
        for line in lines:
            try:
                lines_a.append('\t'.join([line.strip().split('\t')[0], line.strip().split('\t')[1]]))
                lines_a_f.append(line.strip().split('\t')[0])
            except:
                f.write("failed to parse {}".format(line))
                f.write('\n')
    
    index_list = []
    with open(file_b_path, 'r') as fb:
        lines = fb.readlines()
        for line in lines:
            line = line.strip()
            file = line.split('\t')[0]
            num = line.split('\t')[1]
            xpu_status = 'xpu-ops' if file.replace('test/', '').replace('.py', '_xpu.py') in skip_dict.keys() else 'no-xpu-ops'
            xpu_status = '\t' + xpu_status
    
            if file in lines_a_f:
                index = lines_a_f.index(file) 

                xpu_num = lines_a[index].split('\t')[1]
                too_less = "No"
                if int(xpu_num) / int(num) <= 0.2:
                    too_less = "YES"
                too_less = '\t' + too_less

                f.write(lines_a[index] + '\t' + num + xpu_status + too_less)
                index_list.append(index)
            else:
                too_less = "\tYES"
                f.write(file + '\t\t' + num + xpu_status + too_less)

            f.write('\n')
   
    import pdb
    pdb.set_trace()
    with open("xpu.csv") as fa:
        lines = fa.readlines()
        i = 0
        for line in lines:
            if i not in index_list:
                file = lines_a_f[i]
                xpu_status = 'xpu-ops' if file.replace('test/', '') in skip_dict.keys() else 'no-xpu-ops'
                xpu_status = '\t' + xpu_status
                too_less = "\tNo"
                f.write(lines_a[i] + '\t' + xpu_status + too_less)
                f.write('\n') 
            i += 1

import csv
xpu_files = 0
xpu_cases = 0
cuda_files = 0 
cuda_cases = 0
xpu_tested_files = 0
xpu_tested_cases = 0

with open('status.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='#')
    for row in spamreader:
        # Assume xpu-ops test is not upstreamed
        cols = row

        if cols[3] == "xpu-ops":
            xpu_tested_files += 1

        elif cols[1] != "":
            xpu_cases += int(cols[1])
            xpu_tested_cases += int(cols[1])
            if cols[4] == 'No':
                xpu_files += 1
                xpu_tested_files += 1

        if cols[2] != "":
            cuda_files += 1
            cuda_cases += int(cols[2])


print("Cuda tested files    Cuda tested cases   XPU tested files    XPU tested cases    XPU upstremed test files    XPU upstreamed test cases")
print(f"{cuda_files}\t{cuda_cases}\t{xpu_tested_files}\t{xpu_tested_cases}\t{xpu_files}\t{xpu_cases}\n")
            

