#!/usr/bin/env python3
"""
ROS 2 Sync Node based on State Machine Specs.

Flow: Wait for PPS -> Clear State -> Catch All (Cam, Pose, Spec, Radalt)
-> Stamp/Correct -> Save (Multithreaded).
"""

import cv2
import os
import threading
import time
import traceback
import stat
import csv
import yaml
import utm
import glob2
import piexif
import queue
import sqlite3
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
)
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from collections import deque
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Time as BuiltinTime

from PIL import Image as Img
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

# Custom code imports
from .dbConnector import dbConnector
from . import utilities
from .spectral_correct import process_cam0, process_cam1, check_slice_health
from birdseye_msgs.msg import CameraCapture, CaptureComplete

# Tolerant imports — these message types live in repos that may not be
# installed in test/CI containers (inertial_sense_ros2, custom_msgs).
# Subscriptions are skipped when their msg types aren't importable, and
# all_caught() naturally only requires inputs we actually subscribed to.
try:
    from inertial_sense_ros2.msg import DIDINS2
except ImportError:
    DIDINS2 = None
try:
    from ros2_radalt_msgs.msg import AltSNR
except ImportError:
    AltSNR = None
try:
    from as7265x_at_msgs.msg import AS7265xCal
except ImportError:
    AS7265xCal = None

# AS7265x band indices for each cam0 slice — nearest wavelength to each
# camera filter centre. Used to map the 18-band spectrometer to the 4 camera
# bands when computing the per-cycle irradiance ratio correction.
#   slice 0 (450 nm) → index  2 (460 nm)
#   slice 1 (695 nm) → index  9 (705 nm)
#   slice 2 (735 nm) → index 14 (730 nm)
#   slice 3 (850 nm) → index 17 (860 nm)
_CAM0_SPEC_IDX = (2, 9, 14, 17)

# RELIABLE QoS for navigation/sensor data — these topics use RELIABLE
# and must not be dropped (INS, radalt, spectrometer, PPS).
sns_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

# 1. PPS Trigger (The heartbeat of the state machine)
# depth=1: only the latest pulse matters. Prevents sync_node from
# receiving a burst of backlogged PPS messages on startup (which would
# create many simultaneous jobs and flood the drop log).
pps_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


img_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

# TRANSIENT_LOCAL (latched) QoS for the MicaCRPCal panel calibration topic.
# depth=1 so late subscribers always receive the single retained message.
panel_cal_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    depth=1,
)


def deg_to_dms_rational(deg_float):
    """Convert decimal degrees to EXIF-friendly rational DMS."""
    deg = int(deg_float)
    min_float = (deg_float - deg) * 60
    minute = int(min_float)
    sec_float = (min_float - minute) * 60
    sec = int(sec_float * 1000000)
    return [(deg, 1), (minute, 1), (sec, 1000000)]


class RigCalibration:
    def __init__(self, yaml_path):
        with open(yaml_path, "r") as f:
            self.data = yaml.safe_load(f)

    def get_camera_info(self, cam_name):
        cam = self.data["cameras"][cam_name]

        intr = cam["intrinsics"]
        dist = cam["distortion"]
        res = cam["resolution"]
        T_cam_ins = cam["T_cam_ins"]

        K = np.array(
            [
                [intr["fx"], 0.0, intr["cx"]],
                [0.0, intr["fy"], intr["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        D = np.array(
            [dist["k1"], dist["k2"], dist["p1"], dist["p2"], dist.get("k3", 0.0)],
            dtype=np.float64,
        )

        return {
            "K": K,
            "D": D,
            "T_cam_ins": T_cam_ins,
            "width": res["width"],
            "height": res["height"],
        }


class SyncNode(Node):
    def __init__(self):
        super().__init__("sync_node")
        self.br = CvBridge()

        # --- Parameters and Setup ---
        self.declare_parameter("img_format", ".png")
        self.img_format = self.get_parameter("img_format").value

        self.declare_parameter("dir_name", "parsed_flight")
        self.dir_name = self.get_parameter("dir_name").value
        self.dir_name = os.path.join(os.path.expanduser("~"), self.dir_name)
        self.dirCheck()

        # load camera calibration
        self.declare_parameter("calibration_path", "")
        self.calibration_path = os.path.join(
            os.path.expanduser("~"), self.get_parameter("calibration_path").value
        )
        self.calib = RigCalibration(self.calibration_path)
        self.camera_models = {}
        for sensor in ["rgb", "multispec"]:
            for ind in [1, 2, 3, 4]:
                cam_name = f"{sensor}_{ind}"

                cam = self.calib.get_camera_info(cam_name)

                map1, map2 = cv2.initUndistortRectifyMap(
                    cam["K"],
                    cam["D"],
                    None,
                    cam["K"],
                    (cam["width"], cam["height"]),
                    cv2.CV_32FC1,
                )

                self.camera_models[cam_name] = {
                    "cam": cam,
                    "map1": map1,
                    "map2": map2,
                }

        self.spectrometer_wavelengths = [
            410,
            435,
            460,  # ind 2: nearest to 450nm filter
            485,
            510,
            535,
            560,
            585,
            645,
            705,  # ind 9: nearest to 695nm filter
            900,
            940,
            610,
            680,
            730,  # ind 14: nearest to 735nm filter
            760,
            810,
            860,  # ind 17: nearest to 850nm filter
        ]

        db_path = os.path.join(
            os.path.expanduser("~"), self.get_parameter("dir_name").value
        )
        os.makedirs(db_path, exist_ok=True)

        # --- Camera framerate
        self.declare_parameter("framerate", 3.0)
        self.framerate = self.get_parameter("framerate").value

        # --- Ground sample distance (metres per pixel).
        # Update this to match your optics once you have calibration data.
        self.declare_parameter("gsd_m", 0.03)
        self.gsd_m = self.get_parameter("gsd_m").value

        # --- INS Bitmasks ---
        self.HDW_STROBE = 0x00000020
        self.INS_STATUS_SOLUTION_MASK = 0x000F0000
        self.INS_STATUS_SOLUTION_OFFSET = 16
        self.INS_STATUS_GPS_NAV_FIX_MASK = 0x03000000
        self.INS_STATUS_GPS_NAV_FIX_OFFSET = 24
        self.RTK_STATUS = None
        self.INS_STATUS = None

        # --- State Machine Variables ---
        self.state_lock = threading.Lock()
        self.active_jobs = deque(maxlen=10)
        self.max_latency = 0.2  # seconds (tune this)

        # Per-sensor assignment windows (AFTER PPS)
        self.assignment_window = {
            "cam0": 0.95 / self.framerate,
            "cam1": 0.95 / self.framerate,
            "pose": 0.95 / self.framerate,
            "spec": 0.95 / self.framerate,
            "radalt": 0.95 / self.framerate,
        }

        # Allow slight negative offset
        self.pretrigger_tolerance = 0.01

        # --- Subscriptions ---
        self.create_subscription(
            BuiltinTime, "/pps/time", self.pps_cb, qos_profile=pps_qos
        )

        # 2. Camera Streams (BEST_EFFORT to match camera driver publishers)
        self.create_subscription(
            Image, "/cam0/camera_node/image_raw", self.cam0_cb, qos_profile=img_qos
        )
        self.create_subscription(
            Image, "/cam1/camera_node/image_raw", self.cam1_cb, qos_profile=img_qos
        )

        # 3. Navigation & Environment
        if DIDINS2 is not None:
            self.create_subscription(
                DIDINS2, "/ins_quat_uvw_lla", self.ins_cb, qos_profile=sns_qos
            )
        else:
            self.get_logger().warn(
                "inertial_sense_ros2 not available — INS SUB disabled"
            )
        if AltSNR is not None:
            self.create_subscription(
                AltSNR, "/rad_altitude", self.radalt_cb, qos_profile=sns_qos
            )
        else:
            self.get_logger().warn(
                "custom_msgs not available — radar altimeter SUB disabled"
            )

        # 4. AS7265x Spectrometer (For Reflectance)
        if AS7265xCal is not None:
            self.create_subscription(
                AS7265xCal,
                "/as7265x/calibrated_values",
                self.spec_cb,
                qos_profile=sns_qos,
            )
        else:
            self.get_logger().warn(
                "as7265x_at_msgs not available — spectrometer SUB disabled"
            )

        self.capture_pub = self.create_publisher(
            CaptureComplete,
            "/sync/capture_complete",
            10,
        )

        # --- Producer/consumer save queue ---
        self.save_queue = queue.Queue()
        self._save_workers = []
        for _ in range(8):
            t = threading.Thread(target=self._save_worker, daemon=True)
            t.start()
            self._save_workers.append(t)

        threading.Thread(target=self._queue_watchdog, daemon=True).start()
        threading.Thread(target=self._cpu_temp_watchdog, daemon=True).start()

        self.get_logger().info("Sync Node Started. Waiting for PPS Trigger.")

    def dirCheck(self):
        if not os.path.isdir(self.dir_name):
            self.get_logger().info(
                f"{self.dir_name} does not exist in home dir... Generating."
            )
            try:
                os.makedirs(self.dir_name, exist_ok=True)
            except FileExistsError:
                self.get_logger().info(
                    f"{self.dir_name} exists now... Someone beat me to it."
                )
        else:
            self.get_logger().info(f"{self.dir_name} exists...")
            self.clear_dir()
        time.sleep(1)

    def clear_dir(self):
        try:
            files = glob2.glob(os.path.join(self.dir_name, "*"))
            if len(files) >= 1:
                for file in files:
                    if os.path.isfile(file):
                        os.remove(file)
                self.get_logger().info(
                    f"All files in {self.dir_name} deleted successfully.\n"
                )
            else:
                self.get_logger().info(f"No files in {self.dir_name}.\n")
        except Exception as e:
            self.get_logger().info(
                f"Error occurred while clearing {self.dir_name} files: {e}.\n"
            )

    def get_msg_time(self, msg):
        try:
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except AttributeError:
            return time.time()

    def pps_cb(self, msg: BuiltinTime):
        pps_time = msg.sec + msg.nanosec * 1e-9

        job = {
            "pps_time": pps_time,
            "stamp_msg": msg,
            "created_at": time.time(),
            "data": {
                "cam0": None,
                "cam1": None,
                "pose": None,
                "spec": None,
                "radalt": None,
            },
            "dt": {},  # diagnostics
        }

        with self.state_lock:
            self.active_jobs.append(job)

        # Try to resolve older jobs
        self.process_jobs()

    def cam0_cb(self, msg):
        self.assign_to_job("cam0", msg)

    def cam1_cb(self, msg):
        self.assign_to_job("cam1", msg)

    def ins_cb(self, msg):
        # Check if Strobed
        if msg.hdw_status & self.HDW_STROBE == self.HDW_STROBE:
            # self.get_logger().info('    ---> STROBED')
            self.assign_to_job("pose", msg)

    def radalt_cb(self, msg):
        if msg.snr > 13:  # manufacturer-specified SNR floor
            self.assign_to_job("radalt", msg)

    # saves lists in case of a change in msg.values
    def spec_cb(self, msg):
        self.current_raw_spec = list(msg.values)
        self.assign_to_job("spec", msg)

    def _panel_spec_ref_cb(self, msg: Float32MultiArray) -> None:
        self.panel_spec_ref = list(msg.data)
        self.get_logger().info(
            f"Irradiance reference received: {len(self.panel_spec_ref)} bands"
        )
        if self._calibration_ready():
            self.get_logger().info(
                "Calibration complete — image capture and saving now active."
            )

    # --- 3. Processing and Saving ---
    def image_save(self, img, filename, pose):

        # Normalise float32 reflectance → uint16 for all formats.
        if img.dtype == np.float32:
            img = np.clip(img * 65535.0, 0, 65535).astype(np.uint16)

        if self.img_format in (".tiff", ".tif"):
            self._save_geotiff(img, filename, pose)
        elif self.img_format == ".png":
            cv2.imwrite(filename, img)
        elif self.img_format in (".jpeg", ".jpg"):
            self._save_geojpeg(img, filename, pose)

    def _save_geotiff(self, img, filename, pose):
        """Write a georeferenced uint16 GeoTIFF using the INS position."""
        h, w = img.shape[:2]
        bands = 1 if img.ndim == 2 else img.shape[2]

        if pose is not None:
            u = utm.from_latlon(pose.lla[0], pose.lla[1])
            easting, northing, zone_num, zone_letter = u
            is_northern = zone_letter >= "N"
            epsg = 32600 + zone_num if is_northern else 32700 + zone_num
            crs = CRS.from_epsg(epsg)
            # Place image centre at the INS position; derive upper-left corner.
            west = easting - (w / 2.0) * self.gsd_m
            north = northing + (h / 2.0) * self.gsd_m
            transform = from_origin(west, north, self.gsd_m, self.gsd_m)
        else:
            crs = None
            transform = rasterio.transform.IDENTITY

        with rasterio.open(
            filename,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=bands,
            dtype=img.dtype,
            crs=crs,
            transform=transform,
            compress="none",
            photometric="rgb" if bands == 3 else "minisblack",
        ) as dst:
            if bands == 1:
                dst.write(img, 1)
            else:
                for b in range(bands):
                    dst.write(img[:, :, b], b + 1)

    def _save_geojpeg(self, img, filename, pose):
        if img.dtype == np.uint16:
            img = (img >> 8).astype(np.uint8)
        pil_img = Img.fromarray(img)
        if pose is not None:
            lla = pose.lla
            gps_ifd = {
                piexif.GPSIFD.GPSLatitudeRef: "N" if lla[0] >= 0 else "S",
                piexif.GPSIFD.GPSLatitude: deg_to_dms_rational(abs(lla[0])),
                piexif.GPSIFD.GPSLongitudeRef: "E" if lla[1] >= 0 else "W",
                piexif.GPSIFD.GPSLongitude: deg_to_dms_rational(abs(lla[1])),
                piexif.GPSIFD.GPSAltitudeRef: 0,
                piexif.GPSIFD.GPSAltitude: (int(lla[2] * 100), 100),
            }
            exif_bytes = piexif.dump({"GPS": gps_ifd})
            pil_img.save(filename, exif=exif_bytes, format="JPEG", quality=95)
        else:
            pil_img.save(filename, format="JPEG", quality=95)

    def is_complete(self, job):
        return all(v is not None for v in job["data"].values())

    def log_sync_diagnostics(self, job):
        dt_info = job["dt"]
        msg = ", ".join(
            f"{k}:{v * 1000:.1f}ms" for k, v in dt_info.items() if v is not None
        )
        self.get_logger().debug(f"[SYNC] {msg}")

    def assign_to_job(self, key, msg):
        ts = self.get_msg_time(msg)

        with self.state_lock:
            best_job = None
            best_dt = float("inf")

            for job in self.active_jobs:
                dt = ts - job["pps_time"]
                # self.get_logger().info(
                #     f"{key}: ts={ts:.6f}, pps={job['pps_time']:.6f}, dt={dt*1000:.1f}"
                # )
                # Allow small pre-trigger (INS edge case)
                if dt < -self.pretrigger_tolerance:
                    continue

                if dt > self.assignment_window[key]:
                    continue

                abs_dt = abs(dt)

                if abs_dt < best_dt:
                    best_dt = abs_dt
                    best_job = job

            if best_job is None:
                return

            existing = best_job["data"][key]

            if existing is None:
                best_job["data"][key] = msg
                best_job["dt"][key] = best_dt
            else:
                # Replace if closer
                if best_dt < best_job["dt"][key]:
                    best_job["data"][key] = msg
                    best_job["dt"][key] = best_dt

    def process_jobs(self):
        now = time.time()

        with self.state_lock:
            while self.active_jobs:
                job = self.active_jobs[0]

                if now - job["created_at"] < self.max_latency:
                    break  # wait for more data

                self.active_jobs.popleft()

                if self.is_complete(job):
                    self.log_sync_diagnostics(job)
                    self.save_queue.put((job["data"], job["stamp_msg"]))
                else:
                    self.get_logger().warn(
                        f"PPS frame drop @ {job['pps_time']:.3f} (incomplete)"
                    )
                    for key in job["data"].keys():
                        if job["data"][key] is None:
                            self.get_logger().warn(f"    job['data'][{key}] is None")

    def post_process_and_save(self, data, stamp):
        out = CaptureComplete()
        out.header.stamp = stamp
        cams = []
        try:
            # 1. extract data from job
            time_str = f"{stamp.sec}.{str(stamp.nanosec).rjust(9, '0')}"
            self.get_logger().info(f"Saving data frame at timestep {time_str}")

            pose = data["pose"]  # ins_msg
            cam0_raw = self.br.imgmsg_to_cv2(
                data["cam0"], desired_encoding="passthrough"
            )
            cam1_raw = self.br.imgmsg_to_cv2(
                data["cam1"], desired_encoding="passthrough"
            )

            # 2. post process into save/send target formats
            tmp = (pose.ins_status) & self.INS_STATUS_GPS_NAV_FIX_MASK
            RTK_STATUS = tmp >> self.INS_STATUS_GPS_NAV_FIX_OFFSET
            tmp = (pose.ins_status) & self.INS_STATUS_SOLUTION_MASK
            INS_STATUS = tmp >> self.INS_STATUS_SOLUTION_OFFSET

            out.rtk_status = RTK_STATUS
            out.ins_status = INS_STATUS

            spec_for_correction = None

            multispec_cams = process_cam0(cam0_raw, spec_for_correction)  # 4 × (H,W/4)
            for _i, _band in enumerate(multispec_cams):
                for _issue in check_slice_health(_band):
                    self.get_logger().error(f"[IMG HEALTH] cam0[{_i}]: {_issue}")

            rgb_cams = process_cam1(cam1_raw)  # 4 × RGB    (H, W/4, 3)
            for _i, _rgb in enumerate(rgb_cams):
                for _issue in check_slice_health(_rgb):
                    self.get_logger().error(f"[IMG HEALTH] cam1[{_i}]: {_issue}")

            # Convert pose lat-lon -> UTM
            # returns easting, northing, zone number, zone letter
            u = utm.from_latlon(pose.lla[0], pose.lla[1])

            out.utm_letter = u[-1]
            out.utm_number = f"{u[-2]}"

            out.rad_altitude = data["radalt"]

            out.spec_cal = data["spec"]

            t = [  # UTM -> x:easting, y:northing, z:WGS84 altitude
                u[1],  # North
                u[0],  # East
                -pose.lla[2],  # Down
            ]
            quat = [  # quat is scalar-first NED -> convert to scalar-last NED
                pose.qn2b[1],
                pose.qn2b[2],
                pose.qn2b[3],
                pose.qn2b[0],
            ]

            out.ins_pose_ned.position.x = float(t[0])
            out.ins_pose_ned.position.y = float(t[1])
            out.ins_pose_ned.position.z = float(t[2])

            out.ins_pose_ned.orientation.x = float(quat[0])
            out.ins_pose_ned.orientation.y = float(quat[1])
            out.ins_pose_ned.orientation.z = float(quat[2])
            out.ins_pose_ned.orientation.w = float(quat[3])

            # 3. Save Images to File
            fr = 0
            for i, img in enumerate(multispec_cams):
                cam_name = f"multispec_{i+1}"
                cap, filename = self._pack_camera_capture(cam_name, time_str)

                fr += 1
                self.image_save(img, filename, pose)
                cams.append(cap)

            for i, img in enumerate(rgb_cams):
                cam_name = f"rgb_{i+1}"
                cap, filename = self._pack_camera_capture(cam_name, time_str)

                fr += 1
                inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.image_save(inp, filename, pose)
                cams.append(cap)

            self.get_logger().info(
                f"Cycle Complete: Saved {fr} images as {self.img_format} at {time_str}")

            # 4. Send CaptureComplete manifest downstream
            out.cameras = cams
            self.capture_pub.publish(out)

        except Exception as ex:
            self.get_logger().error(
                f"[THREAD] Failed to save: {ex}\n{traceback.format_exc()}"
            )

    def _pack_camera_capture(self, cam_name, time_str, cam_model="pinhole", dist_model="radtan"):
        filename = f"{cam_name}_{time_str}{self.img_format}"
        filepath = os.path.join(self.dir_name, filename)
        cam = self.calib.get_camera_info(cam_name)

        # pack CameraCapture
        cap = CameraCapture()
        cap.camera_name = cam_name
        cap.image_filename = filename
        cap.camera_model = cam_model
        cap.distortion_model = dist_model

        cap.height = cam["height"]
        cap.width = cam["width"]

        cap.fx = cam["K"][0, 0]
        cap.fy = cam["K"][1, 1]
        cap.cx = cam["K"][0, 2]
        cap.cy = cam["K"][1, 2]

        cap.k1 = cam["D"][0]
        cap.k2 = cam["D"][1]
        cap.p1 = cam["D"][2]
        cap.p2 = cam["D"][3]
        cap.k3 = cam["D"][4]

        T_cam_ins = np.array(cam["T_cam_ins"])

        t_cam_ins = T_cam_ins[:3, 3]
        rot_cam_ins = T_cam_ins[:3, :3]

        quat_cam_ins = R.from_matrix(rot_cam_ins).as_quat()

        cap.cam_pose_ins.position.x = float(t_cam_ins[0])
        cap.cam_pose_ins.position.y = float(t_cam_ins[1])
        cap.cam_pose_ins.position.z = float(t_cam_ins[2])

        cap.cam_pose_ins.orientation.x = float(quat_cam_ins[0])
        cap.cam_pose_ins.orientation.y = float(quat_cam_ins[1])
        cap.cam_pose_ins.orientation.z = float(quat_cam_ins[2])
        cap.cam_pose_ins.orientation.w = float(quat_cam_ins[3])

        return cap, filepath

    def _queue_watchdog(self):
        while rclpy.ok():
            sq = self.save_queue.qsize()
            if sq > 0:
                self.get_logger().info(f"Save queue depth: {sq}")
            time.sleep(2.0)

    def _cpu_temp_watchdog(self, warn_c=80.0, crit_c=90.0, interval=10.0):
        while rclpy.ok():
            time.sleep(interval)
            try:
                for zone_path in glob2.glob("/sys/class/thermal/thermal_zone*/temp"):
                    with open(zone_path) as f:
                        temp_c = int(f.read().strip()) / 1000.0
                    zone = zone_path.split("/")[-2]
                    if temp_c >= crit_c:
                        self.get_logger().error(
                            f"[THERMAL] {zone}: {temp_c:.1f}°C — CRITICAL"
                        )
                    elif temp_c >= warn_c:
                        self.get_logger().warn(
                            f"[THERMAL] {zone}: {temp_c:.1f}°C — high"
                        )
            except Exception as e:
                self.get_logger().debug(f"[THERMAL] temp read failed: {e}")

    def _save_worker(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                self.save_queue.task_done()
                break
            data, stamp = item
            try:
                self.post_process_and_save(data, stamp)
            finally:
                self.save_queue.task_done()

    def destroy_node(self):
        # Snapshot depth before sentinels so we don't count them as pending work.
        remaining = self.save_queue.qsize()
        if remaining > 0:
            self.get_logger().info(
                f"Shutdown: waiting for {remaining} queued save(s) to finish..."
            )
            while not self.save_queue.empty():
                self.get_logger().info(
                    f"  save queue: {self.save_queue.qsize()} job(s) remaining"
                )
                time.sleep(2.0)

        for _ in self._save_workers:
            self.save_queue.put(None)

        self.save_queue.join()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SyncNode()
    try:
        while rclpy.ok():
            try:
                rclpy.spin_once(node, timeout_sec=1.0)
            except RuntimeError as e:
                # FastDDS SHM corruption (e.g. after a peer node SIGSEGV) can
                # cause take_message to throw; log and continue rather than
                # crashing the whole node.
                node.get_logger().error(f"Executor RuntimeError (continuing): {e}")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
