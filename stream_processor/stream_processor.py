#!/usr/bin/env python3
"""
ROS 2 Sync Node based on State Machine Specs.

Flow: Wait for PPS -> Clear State -> Catch All (Cam, Pose, Spec, Radalt)
-> Stamp/Correct -> Save (Multithreaded).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time as BuiltinTime

try:
    from inertial_sense_ros2.msg import DIDINS2
except ImportError:
    DIDINS2 = None

import cv2
import os
import threading
import numpy as np
from cv_bridge import CvBridge

import time
import stat
import csv
import yaml
import utm
import glob2
import piexif
from PIL import Image as Img
import concurrent.futures
import copy

# Custom code imports
from .dbConnector import dbConnector
from . import utilities
from .spectral_correct import process_cam0, process_cam1

try:
    from custom_msgs.msg import AltSNR
except ImportError:
    AltSNR = None
try:
    from as7265x_at_msgs.msg import AS7265xCal
except ImportError:
    AS7265xCal = None


# RELIABLE QoS for navigation/sensor data — these topics use RELIABLE publishers
# and must not be dropped (INS, radalt, spectrometer, PPS).
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,  # Reliable delivery
    history=HistoryPolicy.KEEP_LAST,  # Keep last N messages
    depth=10,
)

# BEST_EFFORT QoS for camera images — camera drivers publish with SensorDataQoS
# (BEST_EFFORT). A RELIABLE subscription against a BEST_EFFORT publisher is a
# DDS QoS incompatibility; no messages flow and the mismatch adds protocol overhead.
# For image data, BEST_EFFORT is correct: a missed frame is recovered on the
# next PPS cycle rather than causing backpressure.
img_qos = qos_profile_sensor_data


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

        db_path = os.path.join(os.path.expanduser("~"), self.get_parameter("dir_name").value)
        os.makedirs(db_path, exist_ok=True)
        self.dbc = dbConnector(os.path.join(db_path, self.get_parameter("db_name").value))
        self.dbc.boot(self.get_parameter("db_name").value, self.sensor_id)
        os.chmod(
            os.path.join(self.dir_name, self.db_name + ".db"),
            stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO,
        )
        time.sleep(1)

        # sensor calibration parameters
        self.declare_parameter("sensors_yaml", "sensor_params/birdsEyeSensorParams.yaml")
        self.sensors_yaml = self.get_parameter("sensors_yaml").value
        self.sensors_yaml = os.path.join(os.path.expanduser("~"), self.sensors_yaml)
        self.calibUptake()

        self.declare_parameter("clicks_csv", "catch/data.csv")
        # self.declare_parameter("clicks_csv", "catch/data__2025_01_10.csv")
        self.clicks_csv = self.get_parameter("clicks_csv").value
        self.clicks_csv = os.path.join(os.path.expanduser("~"), self.clicks_csv)
        self.csv_read()

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
        self.current_pps_stamp = None
        # Only populate slots whose message types are importable. all_caught()
        # then naturally requires only the inputs we've actually subscribed to,
        # so the pipeline runs end-to-end in test containers that lack
        # inertial_sense_ros2 / custom_msgs.
        self.caught_data = {"cam0": None, "cam1": None}

        # --- Subscriptions ---
        # 1. PPS Trigger (The heartbeat of the state machine)
        self.create_subscription(BuiltinTime, "/pps/time", self.pps_cb, qos_profile=qos_profile)

        # 2. Camera Streams (BEST_EFFORT to match camera driver publishers)
        self.create_subscription(
            Image, "/cam0/camera_node/image_raw", self.cam0_cb, qos_profile=img_qos
        )
        self.create_subscription(
            Image, "/cam1/camera_node/image_raw", self.cam1_cb, qos_profile=img_qos
        )

        # 3. Navigation & Environment
        if DIDINS2 is not None:
            self.caught_data["pose"] = None
            self.create_subscription(
                DIDINS2, "/ins_quat_uvw_lla", self.ins_cb, qos_profile=qos_profile
            )
        else:
            self.get_logger().warn("inertial_sense_ros2 not available — INS subscription disabled")
        if AltSNR is not None:
            self.caught_data["radalt"] = None
            self.create_subscription(
                AltSNR, "/rad_altitude", self.radalt_cb, qos_profile=qos_profile
            )
        else:
            self.get_logger().warn("custom_msgs not available — radalt subscription disabled")

        # 4. AS7265x Spectrometer (For Reflectance)
        if AS7265xCal is not None:
            self.caught_data["spec"] = None
            self.create_subscription(
                AS7265xCal, "as7265x/calibrated_values", self.spec_cb, qos_profile=qos_profile
            )
        else:
            self.get_logger().warn(
                "as7265x_at_msgs not available — spectrometer subscription disabled"
            )

        self.get_logger().info("Sync Node Initialized. Waiting for PPS Trigger...")

        # --- Multithreading setup ---
        self.save_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def dirCheck(self):
        if not os.path.isdir(self.dir_name):
            self.get_logger().info(f"{self.dir_name} does not exist in home dir... Generating.")
            try:
                os.makedirs(self.dir_name, exist_ok=True)
            except FileExistsError:
                self.get_logger().info(f"{self.dir_name} exists now... Someone beat me to it.")
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
                self.get_logger().info(f"All files in {self.dir_name} deleted successfully.\n")
            else:
                self.get_logger().info(f"No files in {self.dir_name}.\n")
        except Exception as e:
            self.get_logger().info(f"Error occurred while clearing {self.dir_name} files: {e}.\n")

    def csv_read(self):
        self.get_logger().info(f"Reading clicks CSV file: {self.clicks_csv}...")
        data = []
        with open(self.clicks_csv) as clicks:
            reader = csv.reader(clicks)
            for line in reader:
                # breakdown line
                # self.get_logger().info(f'{line}')
                # returns easting, northing, zone number, zone letter
                u = utm.from_latlon(float(line[0]), float(line[1]))
                tag = int(line[-1][-1])
                data.append([u[0], u[1], u[2], u[3], float(line[2]), float(line[3]), tag])
        self.dbc.insertClicks(f"clicks_{self.db_name}", data)
        self.get_logger().info("...Done reading clicks CSV file.\n")

    def calibUptake(self):
        self.get_logger().info(f"Reading sensor parameters YAML file: {self.sensors_yaml}...")
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
                    self.extr = data["T_cam_imu"]  # extrinsics relative to imu base link
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
                    intr2 = [data["gyroscope_noise_density"], data["gyroscope_random_walk"]]
                    self.putParameters(device, res, intr1, intr2, extr)
                elif device == "radalt":
                    extr = data["T_rad_imu"]
                    self.putParameters(
                        device, res, intr1, intr2, utilities.matrix_list_converter(extr, (4, 4))
                    )
                res = None
                intr1 = None
                intr2 = None
                extr = None
        self.get_logger().info("...Done reading sensor parameters YAML file.\n")

    def putParameters(self, device_key, resolution, intrinsics1, intrinsics2, extrinsics):
        vals = '"'
        cols = "sensorID, resolution, intrinsics1, intrinsics2, extrinsics"
        valsList = [device_key, resolution, intrinsics1, intrinsics2, extrinsics]
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

    # --- 1. PPS Trigger (State Machine Start) ---

    def pps_cb(self, msg: BuiltinTime):
        with self.state_lock:
            # Check for "Big Error" logic from diagram:
            # If we get a NEW PPS but haven't "Caught All" from the last one
            if any(v is not None for v in self.caught_data.values()) and not self.all_caught():
                self.get_logger().error(
                    "BIG ERROR: New PPS received before previous cycle completed!"
                )

            # Clear State
            self.current_pps_stamp = msg
            for key in self.caught_data:
                self.caught_data[key] = None
            self.get_logger().info(f"State Cleared. New Sync Cycle: {msg.sec}.{msg.nanosec}")

    # --- 2. Data "Catch" Callbacks ---

    def cam0_cb(self, msg):
        self.catch("cam0", msg)

    def cam1_cb(self, msg):
        self.catch("cam1", msg)

    def ins_cb(self, msg):
        # Check if Strobed
        if msg.hdw_status & self.HDW_STROBE == self.HDW_STROBE:
            self.RTK_STATUS = (
                (msg.ins_status) & self.INS_STATUS_GPS_NAV_FIX_MASK
            ) >> self.INS_STATUS_GPS_NAV_FIX_OFFSET
            self.INS_STATUS = (
                (msg.ins_status) & self.INS_STATUS_SOLUTION_MASK
            ) >> self.INS_STATUS_SOLUTION_OFFSET

            self.catch("pose", msg)

    def radalt_cb(self, msg):
        if msg.snr > 13:
            self.catch("radalt", msg.altitude)

    def spec_cb(self, msg):
        self.catch("spec", msg.values)

    def catch(self, key, data):
        with self.state_lock:
            if self.current_pps_stamp is None:
                return  # Ignore data until first PPS arrives

            if self.caught_data[key] is None:
                self.caught_data[key] = data

            if self.all_caught():
                # Logic: Stamp -> Split/Process -> Save
                self.process_sync_cycle()

    def all_caught(self):
        return all(v is not None for v in self.caught_data.values())

    # --- 3. Processing and Saving ---

    def image_save(self, img, filename, pose):
        if self.img_format in (".tiff", ".tif"):
            # TIFF stores float32 natively — no scaling, full reflectance range
            # preserved for cam0. cv2.imwrite handles uint8/uint16/float32.
            cv2.imwrite(filename, img)
        elif self.img_format == ".png":
            # PNG cannot store float32. Scale to uint16 so the file is viewable;
            # use TIFF if you need to preserve the float32 reflectance range.
            if img.dtype == np.float32:
                img = np.clip(img * 65535.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(filename, img)
        elif self.img_format == ".jpg":
            pil_img = Img.fromarray(img)
            if pose is not None:
                gps_ifd = {
                    piexif.GPSIFD.GPSLatitudeRef: "N" if pose.lla[0] >= 0 else "S",
                    piexif.GPSIFD.GPSLatitude: deg_to_dms_rational(abs(pose.lla[0])),
                    piexif.GPSIFD.GPSLongitudeRef: "E" if pose.lla[1] >= 0 else "W",
                    piexif.GPSIFD.GPSLongitude: deg_to_dms_rational(abs(pose.lla[1])),
                    piexif.GPSIFD.GPSAltitudeRef: 0,
                    piexif.GPSIFD.GPSAltitude: (int(pose.lla[2] * 100), 100),
                }
                exif_bytes = piexif.dump({"GPS": gps_ifd})
                pil_img.save(filename, exif=exif_bytes, format="JPEG", quality=95)
            else:
                pil_img.save(filename, format="JPEG", quality=95)

    def process_sync_cycle(self):
        # Capture a snapshot of data to free the lock quickly
        data = copy.deepcopy(self.caught_data)
        stamp = self.current_pps_stamp

        # Reset state for next cycle immediately
        for key in self.caught_data:
            self.caught_data[key] = None
        self.current_pps_stamp = None

        # Push to save executor for saving
        self.save_executor.submit(self.post_process_and_save, data, stamp)

    def post_process_and_save(self, data, stamp):
        """
        Apply MicaSense-style correction using AS7265x data, then save.

        Writes the per-slice cam0 reflectance and cam1 RGB images to disk and
        records a row in the SQLite DB.
        """
        try:
            pose = data.get("pose")

            # 1. Single-pass split + per-band reflectance (cam0) / debayer (cam1)
            # via the C++ extension. Per-slice cam0 divisor uses CAM0_ALIGNMENT
            # (slice 0->spec[0], 1->spec[2], 2->spec[9], 3->spec[14]).
            cam0_raw = self.br.imgmsg_to_cv2(data["cam0"], desired_encoding="passthrough")
            cam1_raw = self.br.imgmsg_to_cv2(data["cam1"], desired_encoding="passthrough")
            spec_vals = data.get("spec")
            spec_np = np.asarray(spec_vals, dtype=np.float32) if spec_vals is not None else None

            corrected_cam0 = process_cam0(cam0_raw, spec_np)  # 4 × float32 (H, W/4)
            cam1_rgb_list = process_cam1(cam1_raw)  # 4 × RGB    (H, W/4, 3)

            # 2. Save Images to File
            time_str = f"{stamp.sec}.{str(stamp.nanosec).rjust(9,'0')}"
            paths = []
            for i, img in enumerate(corrected_cam0):
                filename = os.path.join(self.dir_name, f"cam0_{i}_{time_str}{self.img_format}")
                paths.append(filename)
                self.image_save(img, filename, pose)

            for i, img in enumerate(cam1_rgb_list):
                filename = os.path.join(self.dir_name, f"cam1_{i}_{time_str}{self.img_format}")
                paths.append(filename)
                self.image_save(img, filename, pose)

            # 3. Save Data Frame to SQL — only when pose (and downstream pose-
            # derived fields) are available. In test environments without
            # inertial_sense_ros2/custom_msgs, skip the DB row but keep images.
            if pose is not None:
                u = utm.from_latlon(pose.lla[0], pose.lla[1])
                paths_str = "|".join(paths)
                vals = [
                    u[0],
                    u[1],
                    pose.lla[2],
                    # quat: scalar-first NED -> scalar-last ENU
                    pose.qn2b[2],
                    pose.qn2b[1],
                    -pose.qn2b[3],
                    pose.qn2b[0],
                    self.RTK_STATUS,
                    self.INS_STATUS,
                    float(data["radalt"]) if data.get("radalt") is not None else 0.0,
                    paths_str,
                    time_str,
                ]
                val_str = ",".join(map(str, vals))
                self.dbc.insertIgnoreInto(
                    f"{self.sensor_id}_images_{self.get_parameter('db_name').value}",
                    "x, y, z, q, u, a, t, rtk_status, ins_status, radalt, save_loc, pps_time",
                    val_str,
                )
            self.get_logger().info(f"Cycle Complete: Saved {len(paths)} at timestep {time_str}")

        except Exception as e:
            self.get_logger().error(f"Post-processing failed: {e}")

    def destroy_node(self):
        self.save_executor.shutdown(wait=True)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
