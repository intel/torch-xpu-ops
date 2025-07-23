file_b_path = "cuda.csv"
lines_a = []
lines_a_f = []
with open("xpu.csv") as fa:
    lines = fa.readlines()
    for line in lines:
        try:
            lines_a.append('\t'.join([line.strip().split('\t')[0], line.strip().split('\t')[1]]))
            lines_a_f.append(line.strip().split('\t')[0])
        except:
            print("failed to parse {}".format(line))

index_list = []
with open(file_b_path, 'r') as fb:
    lines = fb.readlines()
    for line in lines:
        line = line.strip()
        file = line.split('\t')[0]
        num = line.split('\t')[1]
        if file in lines_a_f:
            index = lines_a_f.index(file) 
            print(lines_a[index] + '\t' + str(num))
            index_list.append(index)
        else:
            print(file + '\t\t' + str(num))

with open("xpu.csv") as fa:
    lines = fa.readlines()
    i = 0
    for line in lines:
        if i not in index_list:
            print(lines_a[i])
        i += 1
 
