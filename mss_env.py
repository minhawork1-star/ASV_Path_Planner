"""
mss_env.py — Multi-vehicle MSS (Fossen) wrapper mimicking HoloOcean's API.

v2: Uses REMUS100's built-in `depthHeadingAutopilot` for the AUV inner loop
(realistic, validated control), and Otter's force/RPM mapping for the ASV.

Control input convention (5-element vector, same shape as HoloOcean's TorpedoAUV):
    [0] psi_cmd  — desired yaw (radians, NED)
    [1] z_cmd    — desired depth (meters, +ve down; NED convention)
    [2..4]       — unused (kept for compatibility)
"""

import math
import numpy as np

try:
    from python_vehicle_simulator.vehicles.remus100 import remus100
    from python_vehicle_simulator.vehicles.otter   import otter
    from python_vehicle_simulator.lib.gnc          import attitudeEuler, Rzyx
except ImportError as e:
    raise ImportError(
        "PythonVehicleSimulator not found. Install with:\n"
        "  pip install python-vehicle-simulator\n"
        f"Original error: {e}"
    )

GRAVITY = 9.81


# ============================================================================
class _VehicleWrapper:
    """Wraps one PythonVehicleSimulator vehicle, tracks state, computes IMU."""

    def __init__(self, name, agent_type, x, y, z, yaw_deg, dt,
                 depth_ref=5.0, rpm=1100.0):
        self.name = name
        self.dt   = dt
        self.agent_type = agent_type
        self.depth_ref  = depth_ref
        self.rpm        = rpm

        if agent_type == "TorpedoAUV":
            # REMUS100 with built-in depth+heading autopilot.
            self.vehicle = remus100(
                controlSystem='depthHeadingAutopilot',
                r_z=depth_ref, r_psi=yaw_deg, r_rpm=rpm,
                V_current=0.0, beta_current=0.0,
            )
            # Heading reference-model bandwidth (rad/s).
            self.vehicle.wn_d = 0.6
            self.has_autopilot = True

        elif agent_type == "SurfaceVessel":
            self.vehicle = otter()
            self.has_autopilot = False
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

        # State vectors (Fossen, NED)
        self.eta = np.array([x, y, z, 0.0, 0.0, math.radians(yaw_deg)], dtype=float)
        self.nu  = np.zeros(6, dtype=float)
        if agent_type == "TorpedoAUV":
            self.nu[0] = 1.5  # surge speed at start

        self.u_actual  = np.zeros(self.vehicle.dimU, dtype=float)
        self.u_control = np.zeros(self.vehicle.dimU, dtype=float)
        self.nu_prev   = self.nu.copy()

        # Autopilot setpoints
        self.psi_cmd = math.radians(yaw_deg)
        self.z_cmd   = depth_ref

        # Optional navigator estimate for the inner autopilot. A real AUV's heading/depth loop closes on
        # what its navigation system reports, NOT on ground truth. If these stay None the autopilot falls
        # back to the true state, reproducing the original behaviour bit-for-bit.
        self.eta_est = None
        self.nu_est  = None

    # ----------------------------------------------------------------
    def set_estimate(self, eta, nu):
        """Hand the navigator's estimate (eta_hat, nu_hat) to the inner autopilot for the NEXT step()."""
        self.eta_est = None if eta is None else np.asarray(eta, dtype=float).ravel()
        self.nu_est  = None if nu  is None else np.asarray(nu,  dtype=float).ravel()

    # ----------------------------------------------------------------
    def step(self):
        self.nu_prev = self.nu.copy()

        if self.has_autopilot:
            # REMUS autopilot computes [delta_r, delta_s, n_rpm] from ref_psi, ref_z.
            # It reads z=eta[2], theta=eta[4], psi=eta[5], w=nu[2], q=nu[4], r=nu[5]. Feed it the NAVIGATOR'S
            # estimate when one has been supplied; otherwise fall back to truth (original behaviour).
            eta_ap = self.eta if self.eta_est is None else self.eta_est
            nu_ap  = self.nu  if self.nu_est  is None else self.nu_est
            self.u_control = self.vehicle.depthHeadingAutopilot(
                eta_ap, nu_ap, self.dt
            )

        self.nu, self.u_actual = self.vehicle.dynamics(
            self.eta, self.nu, self.u_actual, self.u_control, self.dt
        )
        self.eta = attitudeEuler(self.eta, self.nu, self.dt)

    # ----------------------------------------------------------------
    def set_control(self, u_input):
        u = np.asarray(u_input, dtype=float).ravel()

        if self.agent_type == "TorpedoAUV":
            psi_cmd = float(u[0])
            z_cmd   = float(u[1])
            self.psi_cmd = psi_cmd
            self.z_cmd   = z_cmd
            # REMUS autopilot expects DEGREES for psi
            self.vehicle.ref_psi = math.degrees(psi_cmd)
            self.vehicle.ref_z   = z_cmd

        elif self.agent_type == "SurfaceVessel":
            f_l, f_r = float(u[0]), float(u[1])
            k_otter = 0.026
            n_l = np.sign(f_l) * math.sqrt(abs(f_l) / k_otter) if abs(f_l) > 1e-6 else 0.0
            n_r = np.sign(f_r) * math.sqrt(abs(f_r) / k_otter) if abs(f_r) > 1e-6 else 0.0
            self.u_control = np.array([np.clip(n_l, -157, 157),
                                       np.clip(n_r, -157, 157)])

    # ----------------------------------------------------------------
    def sensor_dict(self):
        x, y, z, roll, pitch, yaw = self.eta
        u, v, w, p, q, r = self.nu
        R = Rzyx(roll, pitch, yaw)
        v_world = R @ np.array([u, v, w])
        ax, ay, az = self._specific_force()
        gyro = np.array([p, q, r])
        return {
            "LocationSensor": np.array([x, y, z], dtype=float),
            "RotationSensor": np.array([roll, pitch, yaw], dtype=float),
            "VelocitySensor": v_world.astype(float),
            "IMUSensor": np.array([[ax, ay, az], gyro], dtype=float),
        }

    def _specific_force(self):
        u, v, w, p, q, r = self.nu
        u_prev, v_prev, w_prev = self.nu_prev[:3]
        roll, pitch = self.eta[3], self.eta[4]
        dudt = (u - u_prev) / self.dt
        dvdt = (v - v_prev) / self.dt
        dwdt = (w - w_prev) / self.dt
        gx_b = -GRAVITY * math.sin(pitch)
        gy_b =  GRAVITY * math.cos(pitch) * math.sin(roll)
        gz_b =  GRAVITY * math.cos(pitch) * math.cos(roll)
        ax = dudt + q * w - r * v - gx_b
        ay = dvdt + r * u - p * w - gy_b
        az = dwdt + p * v - q * u - gz_b
        return ax, ay, az


# ============================================================================
class MSSEnv:
    def __init__(self, scenario_cfg, depth_ref=5.0, rpm=1100.0):
        self.cfg = scenario_cfg
        self.dt  = 1.0 / scenario_cfg.get("ticks_per_sec", 50)
        self._vehicles = {}
        self._agents_compat = {}
        for ag in scenario_cfg["agents"]:
            name = ag["agent_name"]
            agent_type = ag["agent_type"]
            loc = ag["location"]
            rot = ag.get("rotation", [0, 0, 0])
            yaw_deg = float(rot[2])
            vw = _VehicleWrapper(
                name=name, agent_type=agent_type,
                x=float(loc[0]), y=float(loc[1]), z=float(loc[2]),
                yaw_deg=yaw_deg, dt=self.dt,
                depth_ref=depth_ref, rpm=rpm,
            )
            self._vehicles[name] = vw
            self._agents_compat[name] = _AgentCompat(name)

    @property
    def agents(self):
        return self._agents_compat

    def tick(self):
        st = {}
        for name, vw in self._vehicles.items():
            vw.step()
            st[name] = vw.sensor_dict()
        return st

    def act(self, agent_name, control):
        if agent_name in self._vehicles:
            self._vehicles[agent_name].set_control(control)
        else:
            raise KeyError(f"Unknown agent: {agent_name}")

    def set_estimate(self, agent_name, eta, nu):
        """Supply the navigator's (eta_hat, nu_hat) so the inner autopilot closes on the ESTIMATE, not truth.
        Callers that never invoke this get the original truth-fed behaviour, unchanged."""
        if agent_name in self._vehicles:
            self._vehicles[agent_name].set_estimate(eta, nu)
        else:
            raise KeyError(f"Unknown agent: {agent_name}")


class _AgentCompat:
    def __init__(self, name):
        self.name = name
    def set_control_scheme(self, scheme):
        pass


def make(scenario_cfg, depth_ref=5.0, rpm=1100.0):
    return MSSEnv(scenario_cfg, depth_ref=depth_ref, rpm=rpm)


# ============================================================================
if __name__ == "__main__":
    print("[test] mss_env.py v2 — heading-autopilot smoke test\n")
    cfg = {
        "name": "TEST", "ticks_per_sec": 50,
        "agents": [
            {"agent_name": "auv0", "agent_type": "TorpedoAUV",
             "location": [0, 0, 5], "rotation": [0, 0, 0]},
        ]
    }
    env = make(cfg, depth_ref=5.0, rpm=1100.0)
    print("[test] commanding psi=0, z=5m for 500 ticks (10 s)...")
    for i in range(500):
        env.act("auv0", np.array([0.0, 5.0, 0, 0, 0]))
        st = env.tick()
    s = st["auv0"]
    print(f"  pos={s['LocationSensor']}, yaw={math.degrees(s['RotationSensor'][2]):.2f}°")
    print("\n[test] commanding psi=90° (east turn), 1000 ticks (20 s)...")
    for i in range(1000):
        env.act("auv0", np.array([math.radians(90), 5.0, 0, 0, 0]))
        st = env.tick()
    s = st["auv0"]
    print(f"  pos={s['LocationSensor']}, yaw={math.degrees(s['RotationSensor'][2]):.2f}° (expect ~90°)")
    print(f"  depth={s['LocationSensor'][2]:.2f} m (expect ~5.0 m)")
    print("\n[test] OK if yaw approached 90° and depth held at 5 m.")
