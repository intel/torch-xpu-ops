file_b_path = "status2.csv"
lines_a = []
lines_a_f = []
with open("old.csv") as fa:
    lines = fa.readlines()
    for line in lines:
        try:
            line = line.strip()
            lines_a.append(line)
            lines_a_f.append(line.strip().split('\t')[0])
        except:
            print("failed to parse {}".format(line))
index_list = []
my_list = []
with open("merged.csv", 'w') as fc:
    with open(file_b_path, 'r') as fb:
        lines = fb.readlines()
        fc.write('sep=\'\\t\'\n')
        for line in lines:
            line = line.strip()
            file = line.split('\t')[0]
            if file in lines_a_f:
                index = lines_a_f.index(file)
                fc.write(line + '\t' + '\t'.join(lines_a[index].split('\t')[1:]) + '\tok\n')
            else:
                fc.write(line + '\t\t\t\t\tnew\n')
            my_list.append(file)

     
    i = 0
    for line in lines_a:
        line = line.strip()
        file = line.split('\t')[0]

        if file not in my_list and line.startswith("test/") :
            import os
            if os.path.exists(f"../pytorch/{file}"):
                fc.write(file + '\t\t\t\t\t' + '\t'.join(line.split('\t')[1:]) + '\tnot_in_slow_periodic_xpu\n')
            my_list.append(file)
        i += 1
