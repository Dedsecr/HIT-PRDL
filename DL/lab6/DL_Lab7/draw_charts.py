from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter

f = open('log.txt', 'r')
lines = f.readlines()
f.close()
e_lines = []
for line in lines:
    if 'time' in line:
        e_lines.append(line)

loss_list = [float(line[line.index('loss')+7:line.index(',')]) for line in e_lines]
writer = SummaryWriter(log_dir='logs/train')
for idx, loss in enumerate(loss_list):
    writer.add_scalar('loss', loss, idx + 1)
writer.close()

