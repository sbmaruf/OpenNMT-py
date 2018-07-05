import os, sys, subprocess, numpy as np, time

def count_gpu():

    command = 'nvidia-smi'
    out = (subprocess.check_output(command, shell=True)).decode('utf-8')
    flag = cnt = 0
    out = out.split('\n')
    for line in out:
        line = line.strip()
        if line == "|===============================+======================+======================|":
            flag = 1
            continue
        if line == "":
            break
        if flag == 1:
            cnt += 1
    cnt /= 3
    return int(cnt)



threashhold = 3
consumed_memory = []
limit = 1
tot_gpu = count_gpu()
# print(tot_gpu)

while True:
    gpu_set = set()
    command = 'nvidia-smi'
    out = (subprocess.check_output(command, shell=True)).decode('utf-8')
    out = out.split('\n')
    out.reverse()
    cnt = 0
    for i in out:
        sp = i.split()
        # print(sp)
        if len(sp) > 1:
            if sp[3] == "G":
                continue
            gpu_id = sp[1]
            # print(gpu_id)
            gpu_set.add(int(gpu_id))
        if i == '|=============================================================================|':
            break
        # print(i)
        cnt += 1

    for j in range(tot_gpu):
        if j not in gpu_set:
            print("Runnign on GPU", j)
            command = ''
            (subprocess.check_output(command, shell=True)).decode('utf-8')
            consumed_memory.append((j, 'all'))


