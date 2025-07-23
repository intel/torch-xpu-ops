file_b_path = "periodic.csv"
lines_a = []
lines_a_f = []
with open("slow.csv") as fa:
    lines = fa.readlines()
    for line in lines:
        try:
            line = line.strip()
            print(line)
            lines_a.append('\t'.join([line.strip().split('\t')[0], line.strip().split('\t')[1]]))
            lines_a_f.append(line.strip().split('\t')[0])
        except:
            print("failed to parse {}".format(line))


with open(file_b_path, 'r') as fb:
    lines = fb.readlines()
    for line in lines:
        line = line.strip()
        file = line.split('\t')[0]
        if file not in lines_a_f:
            print(line)

