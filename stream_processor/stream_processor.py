#!/usr/bin/env python3

import cv2
import os
import threading
import sqlite3
import time
import stat
import csv
import yaml
import utm
import glob2
import piexif
import concurrent.futures
# import copy

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from collections import deque
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Time as BuiltinTime
from inertial_sense_ros2.msg import DIDINS2

from PIL import Image as Img

# Custom code imports
from . import dbConnector
from . import utilities
from custom_msgs.msg import AltSNR
from as7265x_at_msgs.msg import AS7265xCal

# Create a custom QoSProfile to prevent message drops
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,  # Reliable delivery
    history=HistoryPolicy.KEEP_ALL,  # Keep all messages
)

CAM0_ALIGNMENT = {
        0: ("BP340_UV", 0),    # 340nm -> Proxy Index 0 (410nm)
        1: ("BP450_Blue", 2),  # 450nm -> Index 2 (460nm)
        2: ("BP695_Red", 9),   # 695nm -> Index 9 (705nm)
        3: ("BP735_Edge", 14)  # 735nm -> Index 14 (730nm)
    }


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
        self.panel_calib = None

        db_path = os.path.join(
            os.path.expanduser("~"), self.get_parameter("dir_name").value
        )
        os.makedirs(db_path, exist_ok=True)
        self.dbc = dbConnector.dbConnector(
            os.path.join(db_path, self.get_parameter("db_name").value)
        )
        self.dbc.boot(self.get_parameter("db_name").value, self.sensor_id)
        os.chmod(
            os.path.join(self.dir_name, self.db_name + ".db"),
            stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO,
        )
        time.sleep(1)

        # # sensor calibration parameters
        # self.declare_parameter("sensors_yaml",
        #    "sensor_params/birdsEyeSensorParams.yaml")
        # self.sensors_yaml = self.get_parameter("sensors_yaml").value
        # self.sensors_yaml = os.path.join(
        #    os.path.expanduser('~'), self.sensors_yaml)
        # self.calibUptake()

        self.declare_parameter("clicks_csv", "catch/data.csv")
        # self.declare_parameter("clicks_csv", "catch/data__2025_01_10.csv")
        self.clicks_csv = self.get_parameter("clicks_csv").value
        self.clicks_csv = os.path.join(
            os.path.expanduser("~"), self.clicks_csv)
        self.csv_read()

        # --- Camera framerate
        self.declare_parameter("framerate", 3.0)
        self.dir_name = self.get_parameter("framerate").value

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
            "cam0": 0.95/self.framerate,
            "cam1": 0.95/self.framerate,
            "pose": 0.95/self.framerate,
            "spec": 0.2,
            "radalt": 0.1,
        }

        # Allow slight negative offset (INS may beat PPS)
        self.pretrigger_tolerance = 0.01

        # --- Subscriptions ---
        # 1. PPS Trigger (The heartbeat of the state machine)
        self.create_subscription(
            BuiltinTime, "/pps/time",
            self.pps_cb, qos_profile=qos_profile
        )

        # 2. Camera Streams
        self.create_subscription(
            Image, "/cam0/camera_node/image_raw",
            self.cam0_cb, qos_profile=qos_profile
        )
        self.create_subscription(
            Image, "/cam1/camera_node/image_raw",
            self.cam1_cb, qos_profile=qos_profile
        )

        # 3. Navigation & Environment
        self.create_subscription(
            DIDINS2, "/ins_quat_uvw_lla",
            self.ins_cb, qos_profile=qos_profile
        )
        self.create_subscription(
            AltSNR, "/rad_altitude",
            self.radalt_cb, qos_profile=qos_profile
        )

        # 4. AS7265x Spectrometer (For Reflectance)
        self.create_subscription(
            AS7265xCal, "/as7265x/calibrated_values",
            self.spec_cb, qos_profile=qos_profile
        )

        self.get_logger().info("Sync Node Started. Waiting for PPS Trigger.")

        # --- Services ---
        self.capture_panel_srv = self.create_service(
            Trigger, "spectrometer/capture_panel", self.capture_panel_callback
        )

        # --- Multithreading setup ---
        self.save_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4)

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
                    [u[0], u[1], u[2], u[3],
                     float(line[2]), float(line[3]), tag]
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
                    self.extr = utilities.matrix_list_converter(
                        self.extr, (4, 4))
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
        """Stores the current spectral reading
        as the white reference (100% reflectance)"""

        if self.current_raw_spec is None:
            response.success = False
            response.message = "No data received from sensor yet."
            return response

        self.panel_calib = list(self.current_raw_spec)
        response.success = True
        response.message = (
            f"Captured panel calibration for {len(self.panel_calib)} bands."
        )
        self.get_logger().info(response.message)
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
        ret = self.dbc.getFrom(cols, table,
                               cond=f'WHERE sensorID = "{device_key}"')
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
                "radalt": None,
            },
            "dt": {}  # diagnostics
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
            tmp = (msg.ins_status) & self.INS_STATUS_GPS_NAV_FIX_MASK
            self.RTK_STATUS = tmp >> self.INS_STATUS_GPS_NAV_FIX_OFFSET
            tmp = (msg.ins_status) & self.INS_STATUS_SOLUTION_MASK
            self.INS_STATUS = tmp >> self.INS_STATUS_SOLUTION_OFFSET

            self.catch("pose", msg)

    def radalt_cb(self, msg):
        if msg.snr > 13:
            self.assign_to_job("radalt", msg)

    # saves lists in case of a change in msg.values
    def spec_cb(self, msg):
        self.current_raw_spec = list(msg.values)
        self.assign_to_job("spec", msg)

    # --- 3. Processing and Saving ---
    def split_camarray(self, img, num_cams=4):
        h, w = img.shape[:2]
        sub_w = w // num_cams
        return [img[:, i * sub_w:(i + 1) * sub_w] for i in range(num_cams)]

    def image_save(self, img, filename, pose):
        if self.img_format == ".png":
            cv2.imwrite(filename, img)
        elif self.img_format == ".jpg":
            # Convert OpenCV BGR (or RGB) NumPy image to PIL RGB
            pil_img = Img.fromarray(img)  # For grayscale or already-RGB

            lla = pose.lla
            gps_ifd = {
                piexif.GPSIFD.GPSLatitudeRef: "N" if lla[0] >= 0 else "S",
                piexif.GPSIFD.GPSLatitude: deg_to_dms_rational(abs(lla[0])),
                piexif.GPSIFD.GPSLongitudeRef: "E" if lla[1] >= 0 else "W",
                piexif.GPSIFD.GPSLongitude: deg_to_dms_rational(abs(lla[1])),
                piexif.GPSIFD.GPSAltitudeRef: 0,
                piexif.GPSIFD.GPSAltitude: (int(lla[2] * 100), 100),
            }

            exif_dict = {"GPS": gps_ifd}
            exif_bytes = piexif.dump(exif_dict)

            # Save directly with EXIF
            pil_img.save(filename, exif=exif_bytes, format="JPEG", quality=95)

    def is_complete(self, job):
        return all(v is not None for v in job["data"].values())

    def log_sync_diagnostics(self, job):
        dt_info = job["dt"]
        msg = ", ".join(
            f"{k}:{v*1000:.1f}ms" for k, v in dt_info.items() if v is not None
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
                    self.log_sync_diagnostics(job)

                    self.save_executor.submit(
                        self.post_process_and_save,
                        job["data"],
                        job["stamp_msg"]
                    )
                else:
                    self.get_logger().warn(
                        f"PPS frame drop @ {job['pps_time']:.3f} (incomplete)"
                    )

    def compute_reflectance(self, spec_vals):
        if self.panel_calib is None:
            return None

        if len(self.panel_calib) != len(spec_vals):
            self.get_logger.warn(
                "Panel calibration length does not match current spectrometer \
                values. \nSkipping reflectance calculation."
            )
            return None

        reflectance = []
        for sample, panel in zip(spec_vals, self.panel_calib):
            # Reflectance = Current / Panel_Reference
            # Assuming the panel is a 1.0 lambertian reflector
            val = (sample / panel) if panel != 0 else 0.0
            reflectance.append(float(val))
        return reflectance

    def post_process_and_save(self, data, stamp):
        """
        Implements MicaSense-style correction using AS7265x data
        before saving to disk and SQL.
        """
        try:
            pose = data["pose"]
            spec = data["spec"]
            time_str = f"{stamp.sec}.{str(stamp.nanosec).rjust(9,'0')}"
            self.get_logger().info(f"Saving data frame at timestep {time_str}")

            # --- Convert ---
            cam0_raw = self.br.imgmsg_to_cv2(
                data["cam0"], desired_encoding="passthrough"
            ).copy()
            cam1_raw = self.br.imgmsg_to_cv2(
                data["cam1"], desired_encoding="passthrough"
            ).copy()

            # --- Split ---
            cam0_list = self.split_camarray(cam0_raw)
            cam1_list = self.split_camarray(cam1_raw)

            # --- Correct cam0 (MONO imagery for multispec) only ---
            corrected_cam0 = []
            for i, sub_img in enumerate(cam0_list):
                filter_name, spec_idx = self.CAM0_ALIGNMENT[i]

                # --- RADIOMETRIC CORRECTION ---
                # digital_counts / relevant_irradiance (from spectrometer)
                if spec and len(spec) > spec_idx:
                    irr = spec[spec_idx]
                else:
                    irr = 1.0

                if irr > 0:
                    # Convert to float for correction, then save as uint8
                    corrected = (sub_img.astype(np.float32) / irr)
                    final_img = np.clip(corrected, 0, 255).astype(np.uint8)
                else:
                    final_img = sub_img
                corrected_cam0.append(final_img)

            # Convert pose lat-lon -> UTM
            # returns easting, northing, zone number, zone letter
            u = utm.from_latlon(pose.lla[0], pose.lla[1])

            # 3. Save Images to File
            paths = []
            for i, img in enumerate(corrected_cam0):
                filename = os.path.join(
                    self.dir_name, f"cam0_{i}_{time_str}.{self.img_format}"
                )
                paths.append(filename)
                self.image_save(img, filename, pose)

            for i, img in enumerate(cam1_list):
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
                filename = os.path.join(
                    self.dir_name, f"cam1_{i}_{time_str}.{self.img_format}"
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
                float(data["radalt"].altitude),
                '"' + paths_str + '"',
                time_str,
            ]
            head = "x, y, z, q, u, a, t, "
            head += "rtk_status, ins_status, radalt, save_loc, pps_time"
            val_str = ",".join(map(str, vals))
            db_path = os.path.join(self.dir_name, self.db_name + ".db")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            insert_query = f"""
            INSERT OR IGNORE INTO {self.sensor_id}_images_{self.db_name}
            ({head})
            VALUES ({val_str});
            """
            cursor.execute(insert_query)
            conn.commit()
            conn.close()

            self.get_logger().info(
                f"Cycle Complete: Saved {len(paths)} images at {time_str}"
            )

        except Exception as ex:
            self.get_logger().info(f"[THREAD] Failed to save: {ex}")

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
