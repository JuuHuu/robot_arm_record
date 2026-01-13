import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/juu/Documents/robot_arm_record/exported/joint_state/wrench_filtered.csv")

plt.figure(figsize=(14,7))

# for name in df["joint_name"].unique():
#     df_j = df[df["joint_name"] == name]
#     plt.plot(df_j["time"], df_j["effort_ma"], label=f"{name}_effort_")
#     plt.plot(df_j["time"], df_j["effort"], label=f"{name}_effort")
    
plt.plot(df["time"], df["fx_lp"],label="fx")


# plt.plot(df["t_center"], df["shoulder_pan_joint_velocity"],label="velocity")
    
plt.xlabel("Time (s)")
plt.ylabel("Joint effort/velocity")
plt.title("Shoulder Pan Joint Effort and Velocity over Time")
plt.legend()
plt.grid(True)
plt.show()
