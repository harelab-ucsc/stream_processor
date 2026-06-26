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
# import copy

import numpy as np

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

# Tolerant imports — these message types live in repos that may not be
# installed in test/CI containers (inertial_sense_ros2, custom_msgs).
# Subscriptions are skipped when their msg types aren't importable, and
# all_caught() naturally only requires inputs we actually subscribed to.
try:
    from inertial_sense_ros2.msg import DIDINS2
except ImportError:
    DIDINS2 = None
try:
    from custom_msgs.msg import AltSNR
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

# BEST_EFFORT QoS for camera images — camera drivers publish with SensorDataQoS
# (BEST_EFFORT). A RELIABLE subscription against a BEST_EFFORT publisher is a
# DDS QoS incompatibility; no messages flow. For image data, BEST_EFFORT is
# correct: a missed frame is recovered on the next PPS cycle.
img_qos = qos_profile_sensor_data

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


class SyncNode(Node):
    def __init__(self):
        super().__init__("sync_node")
        self.br = CvBridge()

        # --- Parameters and Setup ---
        self.declare_parameter("db_name", "flight_data")
        self.db_name = self.get_parameter("db_name").value
        self.declare_parameter("img_format", ".png")
        self.img_format = self.get_parameter("img_format").value
        self.sensor_id = "frc_payload"

        self.declare_parameter("dir_name", "parsed_flight")
        self.dir_name = self.get_parameter("dir_name").value
        self.dir_name = os.path.join(os.path.expanduser("~"), self.dir_name)
        self.dirCheck()

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
        self.current_raw_spec = None
        self.panel_calib = None      # 4 per-band factors from MicaCRPCal
        self.panel_spec_ref = None   # 18-band irradiance reference from AutoCalNode

        self.declare_parameter("require_calibration", True)
        self._require_calibration: bool = (
            self.get_parameter("require_calibration").value
        )

        db_path = os.path.join(
            os.path.expanduser("~"), self.get_parameter("dir_name").value
        )
        os.makedirs(db_path, exist_ok=True)
        self.dbc = dbConnector(
            os.path.join(db_path, self.get_parameter("db_name").value)
        )
        self.dbc.boot(self.get_parameter("db_name").value, self.sensor_id)
        os.chmod(
            os.path.join(self.dir_name, self.db_name + ".db"),
            stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO,
        )
        time.sleep(1)

        # sensor calibration parameters
        # default = "sensor_params/birdsEyeSensorParams.yaml"
        # self.declare_parameter("sensors_yaml", default)
        # self.sensors_yaml = self.get_parameter("sensors_yaml").value
        # self.sensors_yaml = os.path.join(
        #     os.path.expanduser("~"), self.sensors_yaml)
        # self.calibUptake()

        self.declare_parameter("clicks_csv", "catch/data.csv")
        # self.declare_parameter("clicks_csv", "catch/data__2025_01_10.csv")
        self.clicks_csv = self.get_parameter("clicks_csv").value
        self.clicks_csv = os.path.join(os.path.expanduser("~"), self.clicks_csv)
        self.csv_read()

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
            "spec": 0.2,
            "radalt": 0.1,
        }

        # Allow slight negative offset (INS may beat PPS)
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

        # 5. MicaCRPCal panel calibration factors (latched).
        self.create_subscription(
            Float32MultiArray,
            "/panel_cal/irradiance",
            self._panel_cal_cb,
            qos_profile=panel_cal_qos,
        )
        # AutoCalNode irradiance reference — 18-band spectrometer snapshot taken
        # once the drone clears 6 m AGL (above any shadow that may have been on
        # the calibration panel). Used to compute the per-cycle irradiance ratio.
        self.create_subscription(
            Float32MultiArray,
            "/panel_cal/spec_ref",
            self._panel_spec_ref_cb,
            qos_profile=panel_cal_qos,
        )

        self.get_logger().info("Sync Node Started. Waiting for PPS Trigger.")

        # --- Services ---
        self.capture_panel_srv = self.create_service(
            Trigger, "spectrometer/capture_panel", self.capture_panel_callback
        )

        # --- Producer/consumer save queue ---
        self.save_queue = queue.Queue()
        self._save_workers = []
        for _ in range(8):
            t = threading.Thread(target=self._save_worker, daemon=True)
            t.start()
            self._save_workers.append(t)

        # Single serialised DB writer — eliminates concurrent sqlite3 write-lock
        # races that produce "database is locked" errors.
        self._db_queue = queue.Queue()
        self._db_writer_thread = threading.Thread(target=self._db_writer, daemon=True)
        self._db_writer_thread.start()

        threading.Thread(target=self._queue_watchdog, daemon=True).start()
        threading.Thread(target=self._cpu_temp_watchdog, daemon=True).start()
        threading.Thread(target=self._calibration_watchdog, daemon=True).start()

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

    def csv_read(self):
        self.get_logger().info(f"Reading clicks CSV: {self.clicks_csv}...")
        data = []
        with open(self.clicks_csv) as clicks:
            reader = csv.reader(clicks)
            for line in reader:
                # returns easting, northing, zone number, zone letter
                u = utm.from_latlon(float(line[0]), float(line[1]))
                tag = int(line[-1][-1])
                data.append(
                    [u[0], u[1], u[2], u[3], float(line[2]), float(line[3]), tag]
                )
        self.dbc.insertClicks(f"clicks_{self.db_name}", data)
        self.get_logger().info("...Done reading clicks CSV file.\n")

    def calibUptake(self):
        self.get_logger().info(
            f"Reading sensor parameters YAML file: {self.sensors_yaml}..."
        )
        devices = [self.sensor_id, "ins", "radalt"]
        res = None
        intr1 = None
        intr2 = None
        extr = None
        with open(self.sensors_yaml, "r") as f:
            params = yaml.safe_load(f)
            for device in devices:
                data = params[device]
                if device == self.sensor_id:
                    self.res = data["resolution"]
                    self.K = data["intrinsics"]
                    self.dist = data["distortion_coeffs"]
                    self.extr = data["T_cam_imu"]
                    self.extr = utilities.matrix_list_converter(self.extr, (4, 4))
                    res = self.res
                    intr1 = self.K
                    intr2 = self.dist
                    extr = self.extr
                    self.putParameters(device, res, intr1, intr2, extr)
                elif device == "ins":
                    intr1 = [
                        data["accelerometer_noise_density"],
                        data["accelerometer_random_walk"],
                    ]
                    intr2 = [
                        data["gyroscope_noise_density"],
                        data["gyroscope_random_walk"],
                    ]
                    self.putParameters(device, res, intr1, intr2, extr)
                elif device == "radalt":
                    extr = data["T_rad_imu"]
                    self.putParameters(
                        device,
                        res,
                        intr1,
                        intr2,
                        utilities.matrix_list_converter(extr, (4, 4)),
                    )
                res = None
                intr1 = None
                intr2 = None
                extr = None
        self.get_logger().info("  Done reading sensor parameters YAML file.\n")

    # From micasense_spectrometer_bridge.py:
    # Cleaned it up a bit:
    # Add a persistent current_raw_spec buffer so capture_panel_callback
    # can always save the latest
    # spectrometer reading for panel calibration instead of relying on
    # PPS-cycle state that may already have been cleared.
    def capture_panel_callback(self, request, response):
        """Reject the request; use MicaCRPCal panel_scan for panel calibration."""
        response.success = False
        response.message = (
            "This service is deprecated. Run the MicaCRPCal panel_scan node before "
            "flight and hold the CRP panel under cam0 with the QR tag visible."
        )
        self.get_logger().warn(response.message)
        return response

    def putParameters(self, dev_key, res, intr1, intr2, extr):
        vals = '"'
        cols = "sensorID, resolution, intrinsics1, intrinsics2, extrinsics"
        valsList = [dev_key, res, intr1, intr2, extr]
        vals += '","'.join([str(x) for x in valsList])
        vals += '"'
        self.dbc.insertIgnoreInto(f"parameters_{self.db_name}", cols, vals)

    def getParameters(self, device_key):
        params = []
        cols = "sensorID, resolution, intrinsics1, intrinsics2, extrinsics"
        table = f"parameters_{self.db_name}"
        ret = self.dbc.getFrom(cols, table, cond=f'WHERE sensorID = "{device_key}"')
        for elem in ret:
            for i, item in enumerate(elem):
                if item == device_key:
                    params.append(item)
                elif item != "None":
                    tmp = utilities.string_list_converter(item)
                    if item == elem[-1]:
                        tmp = utilities.matrix_list_converter(tmp, (4, 4))
                    params.append(tmp)
        return params

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
                # "radalt": None,
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
            tmp = (msg.ins_status) & self.INS_STATUS_GPS_NAV_FIX_MASK
            self.RTK_STATUS = tmp >> self.INS_STATUS_GPS_NAV_FIX_OFFSET
            tmp = (msg.ins_status) & self.INS_STATUS_SOLUTION_MASK
            self.INS_STATUS = tmp >> self.INS_STATUS_SOLUTION_OFFSET

            self.assign_to_job("pose", msg)

    def radalt_cb(self, msg):
        if msg.snr > 13:
            self.assign_to_job("radalt", msg)

    # saves lists in case of a change in msg.values
    def spec_cb(self, msg):
        self.current_raw_spec = list(msg.values)
        self.assign_to_job("spec", msg)

    def _panel_cal_cb(self, msg: Float32MultiArray) -> None:
        self.panel_calib = list(msg.data)
        self.get_logger().info(
            f"Panel calibration received: {len(self.panel_calib)} band factors"
        )

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
        elif self.img_format == ".jpg":
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

    def is_complete(self, job):
        return all(v is not None for v in job["data"].values())

    def _calibration_ready(self) -> bool:
        if not self._require_calibration:
            return True
        return self.panel_calib is not None and self.panel_spec_ref is not None

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
                    if not self._calibration_ready():
                        self.get_logger().warn(
                            "Calibration not yet complete — discarding image cycle. "
                            "Waiting for /panel_cal/irradiance (panel scan) "
                            "and /panel_cal/spec_ref (auto_cal at 6 m AGL).",
                            throttle_duration_sec=10.0,
                        )
                    else:
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
        """Apply per-band reflectance correction and save images + DB record."""
        try:
            pose = data["pose"]
            radalt = data["radalt"].altitude
            time_str = f"{stamp.sec}.{str(stamp.nanosec).rjust(9, '0')}"
            self.get_logger().info(f"Saving data frame at timestep {time_str}")

            # Extract current spectrometer values for the irradiance ratio correction.
            _spec_msg = data["spec"]
            spec_np = np.asarray(
                _spec_msg if isinstance(_spec_msg, np.ndarray) else _spec_msg.values,
                dtype=np.float32,
            )

            # Convert images once
            cam0_raw = self.br.imgmsg_to_cv2(
                data["cam0"], desired_encoding="passthrough"
            )
            cam1_raw = self.br.imgmsg_to_cv2(
                data["cam1"], desired_encoding="passthrough"
            )

            if not self._require_calibration and self.panel_calib is None:
                # Test / force_cal mode with no calibration data — skip correction.
                spec_for_correction = None
            elif not self._calibration_ready():
                # Should not reach here — process_jobs() gates on _calibration_ready().
                raise RuntimeError(
                    "post_process_and_save called before calibration is ready"
                )
            elif self.panel_spec_ref is None:
                # panel_calib present but no irradiance reference yet (shouldn't
                # happen in normal flight; panel_spec_ref arrives with auto_cal).
                spec_for_correction = np.asarray(self.panel_calib, dtype=np.float32)
            else:
                # total_factor[i] = panel_factor[i] * (spec_ref[k] / spec_current[k])
                # Corrects for irradiance changes relative to the reference captured
                # when the drone first cleared 6 m AGL.
                total = []
                for i, k in enumerate(_CAM0_SPEC_IDX):
                    cur = float(spec_np[k])
                    irr_ratio = float(self.panel_spec_ref[k]) / cur if cur > 0.0 else 1.0
                    total.append(float(self.panel_calib[i]) * irr_ratio)
                spec_for_correction = np.asarray(total, dtype=np.float32)

            corrected_cam0 = process_cam0(cam0_raw, spec_for_correction)  # 4 × (H,W/4)
            for _i, _band in enumerate(corrected_cam0):
                for _issue in check_slice_health(_band):
                    self.get_logger().error(f"[IMG HEALTH] cam0[{_i}]: {_issue}")

            cam1_rgb_list = process_cam1(cam1_raw)  # 4 × RGB    (H, W/4, 3)
            for _i, _rgb in enumerate(cam1_rgb_list):
                for _issue in check_slice_health(_rgb):
                    self.get_logger().error(f"[IMG HEALTH] cam1[{_i}]: {_issue}")

            # Convert pose lat-lon -> UTM
            # returns easting, northing, zone number, zone letter
            u = utm.from_latlon(pose.lla[0], pose.lla[1])

            # 3. Save Images to File
            paths = []
            for i, img in enumerate(corrected_cam0):
                filename = os.path.join(
                    self.dir_name, f"multispec_{i}_{time_str}{self.img_format}"
                )
                paths.append(filename)
                self.image_save(img, filename, pose)

            for i, img in enumerate(cam1_rgb_list):
                filename = os.path.join(
                    self.dir_name, f"rgb_{i}_{time_str}{self.img_format}"
                )
                paths.append(filename)
                self.image_save(img, filename, pose)

            # 4. Save Data Frame to SQL
            # Format: x, y, z, q, u, a, t, status, radalt, path, time...
            paths_str = "|".join(paths)
            vals = [
                # UTM -> save x:easting, y:northing, z:WGS84 altitude
                u[0],
                u[1],
                pose.lla[2],
                # quat is scalar-first NED -> convert to scalar-last NED
                pose.qn2b[1],
                pose.qn2b[2],
                pose.qn2b[3],
                pose.qn2b[0],
                self.RTK_STATUS,
                self.INS_STATUS,
                float(radalt),
                '"' + paths_str + '"',
                time_str,
            ]
            head = "x, y, z, q, u, a, t, "
            head += "rtk_status, ins_status, radalt, save_loc, pps_time"
            val_str = ",".join(map(str, vals))
            self._db_queue.put((head, val_str))

            self.get_logger().info(
                f"Cycle Complete: Saved {len(paths)} images at {time_str}"
            )

        except Exception as ex:
            self.get_logger().error(
                f"[THREAD] Failed to save: {ex}\n{traceback.format_exc()}"
            )

    def _calibration_watchdog(self, timeout: float = 60.0, repeat: float = 30.0) -> None:
        """Log a loud error if calibration hasn't arrived after `timeout` seconds."""
        time.sleep(timeout)
        if self._calibration_ready():
            return
        sep = "=" * 62
        while not self._calibration_ready():
            missing = []
            if self.panel_calib is None:
                missing.append("/panel_cal/irradiance (panel_scan node)")
            if self.panel_spec_ref is None:
                missing.append("/panel_cal/spec_ref  (auto_cal node)")
            self.get_logger().error(
                f"\n{sep}\n"
                f"  CALIBRATION NOT RECEIVED after {timeout:.0f} s\n"
                f"  Still waiting for:\n"
                + "".join(f"    - {m}\n" for m in missing)
                + f"  Images are being DISCARDED until calibration arrives.\n"
                f"  Check that mica_crp_cal is built and auto_cal/panel_scan\n"
                f"  nodes are running (`ros2 node list | grep cal`).\n"
                f"{sep}"
            )
            time.sleep(repeat)

    def _queue_watchdog(self):
        while rclpy.ok():
            sq = self.save_queue.qsize()
            dq = self._db_queue.qsize()
            if sq > 0 or dq > 0:
                self.get_logger().info(f"Save queue depth: {sq}  DB queue depth: {dq}")
            time.sleep(2.0)

    def _db_writer(self):
        db_path = os.path.join(self.dir_name, self.db_name + ".db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        cursor = conn.cursor()
        while True:
            item = self._db_queue.get()
            if item is None:
                self._db_queue.task_done()
                break
            head, val_str = item
            try:
                cursor.execute(
                    f"INSERT OR IGNORE INTO {self.sensor_id}_images_{self.db_name}"
                    f" ({head}) VALUES ({val_str});"
                )
                conn.commit()
            except Exception as e:
                self.get_logger().error(f"[DB] Write failed: {e}")
            finally:
                self._db_queue.task_done()
        conn.close()

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

        # Drain the DB writer after all image workers have flushed their inserts.
        self._db_queue.put(None)
        self._db_queue.join()

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
