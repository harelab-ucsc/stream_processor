#!/usr/bin/env python3

import sqlite3
import math

import pandas as pd


def bytesIOconverter(bytes_io_object):
    return bytes_io_object.getvalue()


class dbConnector:
    def __init__(self, db_name) -> None:
        # Converts np.array to TEXT when inserting
        # sqlite3.register_adapter(np.ndarray, self.adapt_array)
        # Converts TEXT to np.array when selecting
        # sqlite3.register_converter("array", self.convert_array)
        self.db_c = sqlite3.connect(f"{db_name}.db", detect_types=sqlite3.PARSE_DECLTYPES)
        self.db_c.create_function("sqrt", 1, math.sqrt)
        self.db_c.create_function("pow", 2, self.sqlite_power)

    def boot(self, db_name, sensor):
        # print(db_name, sensor)
        self.setupTable(
            f"{sensor}_images_{db_name}",
            "x REAL, y REAL, z REAL, q REAL, u REAL, a REAL, t REAL, "
            "rtk_status INTEGER, ins_status INTEGER, radalt REAL, "
            "save_loc TEXT UNIQUE, cam_time1 REAL, cam_time2 REAL, "
            "ins_time1 REAL, ins_time2 REAL",
        )
        self.setupTable(
            f"clicks_{db_name}",
            "x REAL, y REAL, zone_num INTEGER, zone_letter TEXT, "
            "z REAL, z_msl REAL, tag INTEGER",
        )
        self.setupTable(
            f"parameters_{db_name}",
            "sensorID TEXT UNIQUE, resolution array, intrinsics1 array, "
            "intrinsics2 array, extrinsics array",
        )
        self.setupTable(
            f"ins_data_{db_name}",
            "x REAL, y REAL, z REAL, q REAL, u REAL, a REAL, t REAL, "
            "insStatus INTEGER, hdwStatus INTEGER, "
            "time1 REAL UNIQUE, time2 REAL",
        )

    def sqlite_power(self, x, n):
        return int(x) ** n

    def checkForTable(self, table_name):
        cur = self.db_c.cursor()
        res = cur.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        )
        if len(res.fetchall()) == 0:
            return False
        return True

    def diagnostic(self, table, max=0):
        cur = self.db_c.cursor()
        limit = ";"
        if max != 0:
            limit = f" LIMIT {max};"
        res = cur.execute(f"SELECT * FROM {table}" + limit)
        print(res.fetchall())

    def getFrom(self, what, where, max=0, cond=None):
        cur = self.db_c.cursor()
        limit = ";"
        if max != 0:
            limit = f" LIMIT {max};"
        if cond is None:
            cond = " "
        query = f"SELECT {what} FROM {where}" + " " + cond + limit
        # print('dbc: ', query)
        res = cur.execute(query)
        return res.fetchall()  # list of all rows of query result

    def dfToTable(self, data, where, over_write=True):
        proc_d = self.checkForTable(where)
        if proc_d and over_write:
            cur = self.db_c.cursor()
            cur.execute(f"DROP TABLE {where}")
            data.to_sql(name=where, con=self.db_c)
        if not proc_d:
            data.to_sql(name=where, con=self.db_c)

    def tableToDF(self, where):
        return pd.read_sql_query(f"SELECT * FROM {where}", self.db_c)

    def setupTable(self, table_name, cols):
        cur = self.db_c.cursor()
        res = cur.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        )
        if len(res.fetchall()) == 0:
            cur.execute(f"CREATE TABLE {table_name}({cols})")
            self.db_c.commit()

    def insertInto(self, table_name, cols, vals):
        cur = self.db_c.cursor()
        cur.execute(f"INSERT INTO {table_name}({cols}) VALUES({vals})")
        self.db_c.commit()

    def insertIgnoreInto(self, table_name, cols, vals):
        cur = self.db_c.cursor()
        cur.execute(f"INSERT OR IGNORE INTO {table_name}({cols}) VALUES({vals})")
        self.db_c.commit()

    def insertClicks(self, table_name, vals):
        cur = self.db_c.cursor()
        # cur.executemany(f"INSERT INTO {table_name} (x, y, health) VALUES(?,?,?)", vals)
        cur.executemany(
            f"INSERT INTO {table_name} "
            "(x, y, zone_num, zone_letter, z, z_msl, tag) VALUES(?,?,?,?,?,?,?)",
            vals,
        )
        self.db_c.commit()

    def updateDataDetections(self, table_name, vals):
        cur = self.db_c.cursor()
        cur.executemany(
            f"UPDATE {table_name} SET blob_center_x = ?, blob_center_y = ? WHERE img_loc = ?;",
            vals,
        )
        self.db_c.commit()

    def dropTable(self, table_name):
        cur = self.db_c.cursor()
        # check if table exists
        if not self.checkForTable(table_name):
            return
        cur.execute(f"DROP TABLE {table_name}")
        self.db_c.commit()

    def insertMany(self, table_name, cols, vals):
        cur = self.db_c.cursor()
        num_columns = len(cols.split(","))
        # Build the SQL statement to insert rows
        placeholders = ",".join(["?" for _ in range(num_columns)])
        cur.executemany(f"INSERT INTO {table_name}({cols}) VALUES({placeholders})", vals)
        self.db_c.commit()
