#!/usr/bin/env python3

import os
import math

import pandas as pd

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


BAG_PATH = "/home/juu/Documents/robot_arm_record/Original_data/apply_force_04_00"  # folder that contains metadata.yaml and .db3
EXPORTED_CSV_DIR = "exported/apply_force_04_00"


def open_reader(bag_path: str) -> SequentialReader:
    storage_options = StorageOptions(
        uri=bag_path,
        storage_id="sqlite3",
    )
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def build_topic_type_map(reader: SequentialReader):
    """Get mapping: topic_name -> (type_name, msg_class)."""
    topic_info_list = reader.get_all_topics_and_types()
    topic_type_map = {}
    for t in topic_info_list:
        # t.name, t.type
        msg_cls = get_message(t.type)
        topic_type_map[t.name] = (t.type, msg_cls)
    return topic_type_map


def main():
    reader = open_reader(BAG_PATH)
    topic_type_map = build_topic_type_map(reader)

    print("Topics in bag:")
    for name, (type_name, _) in topic_type_map.items():
        print(f"  {name} : {type_name}")

    # Storage for rows
    joint_rows = []
    wrench_rows = []

    # Main read loop
    while reader.has_next():
        topic, data, t = reader.read_next()  # t is int nanoseconds
        t_sec = t / 1e9

        if topic not in topic_type_map:
            continue

        _, msg_cls = topic_type_map[topic]
        msg = deserialize_message(data, msg_cls)

        # /joint_states
        if topic == "/joint_states":
            # sensor_msgs/msg/JointState
            # msg.name, msg.position, msg.velocity, msg.effort are arrays
            n = len(msg.name)
            for i in range(n):
                name = msg.name[i]

                # Safe indexing
                pos = msg.position[i] if i < len(msg.position) else math.nan
                vel = msg.velocity[i] if i < len(msg.velocity) else math.nan
                eff = msg.effort[i] if i < len(msg.effort) else math.nan

                joint_rows.append({
                    "time": t_sec,
                    "joint_name": name,
                    "position": pos,
                    "velocity": vel,
                    "effort": eff,
                })

        # /force_torque_sensor_broadcaster/wrench
        elif topic == "/force_torque_sensor_broadcaster/wrench":
            # geometry_msgs/msg/WrenchStamped
            w = msg.wrench
            wrench_rows.append({
                "time": t_sec,
                "fx": w.force.x,
                "fy": w.force.y,
                "fz": w.force.z,
                "tx": w.torque.x,
                "ty": w.torque.y,
                "tz": w.torque.z,
            })

    # Convert to DataFrame and save
    os.makedirs(f"{EXPORTED_CSV_DIR}", exist_ok=True)

    if joint_rows:
        df_joint = pd.DataFrame(joint_rows)
        df_joint.to_csv(f"{EXPORTED_CSV_DIR}/joint_states.csv", index=False)
        print(f"Saved {len(df_joint)} joint_state rows to {EXPORTED_CSV_DIR}/joint_states.csv")
    else:
        print("No joint_state messages found!")

    if wrench_rows:
        df_wrench = pd.DataFrame(wrench_rows)
        df_wrench.to_csv(f"{EXPORTED_CSV_DIR}/wrench.csv", index=False)
        print(f"Saved {len(df_wrench)} wrench rows to {EXPORTED_CSV_DIR}/wrench.csv")
    else:
        print("No wrench messages found!")


if __name__ == "__main__":
    main()

