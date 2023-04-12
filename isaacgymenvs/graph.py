import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

# data_dir = "/home/jkim3662/Downloads/"

# ours_command_x = np.genfromtxt(data_dir+"ours_command_x.csv",delimiter=',')[1:,2]
# ours_rope_x = np.genfromtxt(data_dir+"ours_rope_x.csv",delimiter=',')[1:,2]
# steps = np.genfromtxt(data_dir+"ours_command_x.csv",delimiter=',')[1:,1]

# ours_command_y = np.genfromtxt(data_dir+"ours_command_y.csv",delimiter=',')[1:,2]
# ours_rope_y = np.genfromtxt(data_dir+"ours_rope_y.csv",delimiter=',')[1:,2]

# ours_command_z = np.genfromtxt(data_dir+"ours_command_z.csv",delimiter=',')[1:,2]
# ours_rope_z = np.genfromtxt(data_dir+"ours_rope_z.csv",delimiter=',')[1:,2]


# ours_command = np.vstack((ours_command_x,ours_command_y,ours_command_z))
# ours_rope = np.vstack((ours_rope_x,ours_rope_y,ours_rope_z))
# ours_err = np.linalg.norm(ours_command-ours_rope,axis=0)
# ours_reward = np.exp(-ours_err**2)
# plt.plot(steps[::10], ours_reward[::10])
# plt.show()

log_dir = "/home/jkim3662/Projects/Rope/IsaacGymEnvs/isaacgymenvs/runs/FrankaRope/summaries/policy/"
event_acc = event_accumulator.EventAccumulator(log_dir)
event_acc.Reload()

tags = event_acc.Tags()["scalars"]
data = {}
for tag in tags:
    if tag[:3] != "obs" and tag[:3] != "rew":
        scalar_data = event_acc.Scalars(tag)
        itr = 0
        for scalar_datum in scalar_data:
            if itr == 0:
                data[tag] = []
                steps = []
            data[tag].append(scalar_datum.value)
            itr+=1
            steps.append(scalar_datum.step)
        plt.plot(steps,data[tag],label=tag)

plt.legend()
plt.show()



