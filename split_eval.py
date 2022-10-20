import sys

gen_list = []
tar_list = []
with open(sys.argv[1]) as f_in:
    for line in f_in:
        q, gen, tar = line.strip().split('\t')
        gen_list.append(gen)
        tar_list.append(tar)
        
with open('output.txt', 'w') as f_out:
    f_out.write('\n'.join(gen_list))
    
with open('target.txt', 'w') as f_out:
    f_out.write('\n'.join(tar_list))