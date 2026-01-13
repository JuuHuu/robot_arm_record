# pip install git+https://github.com/UniversalRobots/RTDE_Python_Client_Library.git

import argparse
import time
from collections import deque

import matplotlib.pyplot as plt

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

XML_CONFIG = """<?xml version="1.0"?>
<rtde_config>
  <recipe key="out">
    <field name="timestamp" type="DOUBLE"/>
    <field name="wrench_calc_from_currents" type="VECTOR6D"/>
  </recipe>
</rtde_config>
"""

LABELS = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]


def build_connection(host: str, port: int, hz: float):
    # Write the XML config to a temp file (most compatible across versions)
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".xml") as f:
        f.write(XML_CONFIG)
        cfg_path = f.name

    conf = rtde_config.ConfigFile(cfg_path)
    os.unlink(cfg_path)

    out_names, out_types = conf.get_recipe("out")

    con = rtde.RTDE(host, port)
    con.connect()
    con.negotiate_protocol_version()

    if not con.send_output_setup(out_names, out_types, frequency=hz):
        con.disconnect()
        raise RuntimeError(
            "Failed to set up RTDE outputs. "
            "Either the variable is NOT_FOUND on your robot, or the recipe is invalid."
        )

    if not con.send_start():
        con.disconnect()
        raise RuntimeError("Failed to start RTDE stream.")

    return con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.100.162")
    ap.add_argument("--port", type=int, default=30004)
    ap.add_argument("--hz", type=float, default=125.0)
    ap.add_argument("--window", type=float, default=5.0, help="rolling window in seconds")
    args = ap.parse_args()

    con = build_connection(args.host, args.port, args.hz)
    maxlen = max(10, int(args.window * args.hz))

    # Rolling buffers
    t_buf = deque(maxlen=maxlen)
    y_buf = [deque(maxlen=maxlen) for _ in range(6)]

    # Matplotlib setup
    plt.ion()
    fig, ax = plt.subplots()
    lines = [ax.plot([], [], label=LABELS[i])[0] for i in range(6)]
    ax.set_title("wrench_calc_from_currents (rolling window)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wrench")
    ax.legend(loc="upper right")

    t0 = None
    last_redraw = time.time()
    redraw_period = 0.05  # seconds (20 Hz UI refresh)

    try:
        while True:
            state = con.receive()
            if state is None:
                continue

            ts = state.timestamp
            wrench = state.wrench_calc_from_currents  # [Fx,Fy,Fz,Tx,Ty,Tz]

            if t0 is None:
                t0 = ts
            t = ts - t0

            t_buf.append(t)
            for i in range(6):
                y_buf[i].append(wrench[i])

            # Redraw at a UI-friendly rate (don’t try to redraw 125 times/sec)
            now = time.time()
            if now - last_redraw >= redraw_period and len(t_buf) >= 2:
                t_list = list(t_buf)
                for i in range(6):
                    lines[i].set_data(t_list, list(y_buf[i]))

                ax.set_xlim(t_list[0], t_list[-1])

                # autoscale y based on current data window
                all_y = [v for buf in y_buf for v in buf]
                y_min, y_max = min(all_y), max(all_y)
                if y_min == y_max:
                    y_min -= 1.0
                    y_max += 1.0
                pad = 0.05 * (y_max - y_min)
                ax.set_ylim(y_min - pad, y_max + pad)

                fig.canvas.draw()
                fig.canvas.flush_events()
                last_redraw = now

    except KeyboardInterrupt:
        pass
    finally:
        try:
            con.send_pause()
        except Exception:
            pass
        con.disconnect()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()