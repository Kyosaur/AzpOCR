import datetime
import os
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytesseract
import sqlite3

from dateutil.parser import parse
from pytesseract.pytesseract import Output
from pathlib import Path


class AzpOCR:
    MASTER_FILE = "Directory.db"

    TABLE_NAME = "directory"
    CONVERSION_TABLE = "conversions"

    SORTED_FOLDER = "Sorted"
    CONVERTED_FOLDER = "Converted"
    GIVEUP_FOLDER = "Not Sorted"

    def __init__(self, start_path) -> None:
        self.starting_path = start_path
        self._create_main_db(start_path)
        self._create_conversion_db()

    def _query_db(self, query, commit=False):
        with sqlite3.connect(self.starting_path + self.MASTER_FILE) as conn:
            ret = conn.execute(query)
            if commit:
                conn.commit()
        return ret

    def _create_main_db(self, location):
        query = f"""CREATE TABLE IF NOT EXISTS "{self.TABLE_NAME}" (
                "id"	INTEGER UNIQUE,
                "file_location"	TEXT,
                "file_extension"	INTEGER,
                "can_read"      INTEGER DEFAULT 1,
                "was_scanned"	INTEGER DEFAULT 0,
                "scan_success"	INTEGER DEFAULT 0,
                "scan_attempts" INTEGER DEFAULT 0,
                "date_pulled"	TEXT,
                "new_location"	TEXT,
                "was_converted" INTEGER DEFAULT 0,
                PRIMARY KEY("id" AUTOINCREMENT)
            );"""
        self._query_db(query, True)

    def _create_conversion_db(self):
        query = f"""CREATE TABLE IF NOT EXISTS "{self.CONVERSION_TABLE}" (
                "id"	INTEGER UNIQUE,
                "file_location"	TEXT,
                "file_extension" TEXT,
                "was_converted" INTEGER DEFAULT 0,
                "convert_attempts" INTEGER DEFAULT 0,
                "new_location"	TEXT,
                "new_extension" TEXT,
                PRIMARY KEY("id" AUTOINCREMENT)
            );"""
        self._query_db(query, commit=True)

    def _push_converted_files(self):
        query = f"""UPDATE `{self.TABLE_NAME}` SET `was_converted`=1 FROM `{self.CONVERSION_TABLE}` 
                WHERE {self.TABLE_NAME}.file_location = {self.CONVERSION_TABLE}.file_location 
                AND {self.CONVERSION_TABLE}.was_converted=1"""
        self._query_db(query, commit=True)

    def _populate_conversion_table(self, ext1, ext2):
        query = f"""INSERT INTO `{self.CONVERSION_TABLE}` (`file_location`, `file_extension`)
                SELECT `file_location`, `file_extension` FROM `{self.TABLE_NAME}`
                WHERE `file_extension` = '{ext1}'
                AND NOT EXISTS (
                    SELECT 1
                    FROM `{self.CONVERSION_TABLE}`
                    WHERE {self.CONVERSION_TABLE}.file_location = {self.TABLE_NAME}.file_location
                );"""
        self._query_db(query, commit=True)

    def _file_not_readable(self, file):
        query = f"""UPDATE `{self.TABLE_NAME}` SET `can_read`=0, 
            `scan_attempts`= (SELECT `scan_attempts` FROM `{self.TABLE_NAME}`
            WHERE `file_location`='{file}')+1
            WHERE `file_location`='{file}';"""
        self._query_db(query, commit=True)

    def _date_found(self, file, date):
        new_location = (
            self.starting_path
            + f"{self.SORTED_FOLDER}\\"
            + str(date.year)
            + "\\"
            + str(date.strftime("%B"))
            + "\\"
        )
        query = f"""UPDATE `{self.TABLE_NAME}` SET `can_read`=1, `was_scanned`=1, 
                `scan_success`=1, `date_pulled`='{date.strftime("%m/%d/%Y")}', 
                `new_location`='{new_location + Path(file).name}',
                `scan_attempts`= (SELECT `scan_attempts` FROM `{self.TABLE_NAME}`
                WHERE `file_location`='{file}')+1
                WHERE `file_location`='{file}';"""

        self._query_db(query, commit=True)
        self._copy_file(file, new_location)

    def _no_date_found(self, file):
        query = f"""UPDATE `{self.TABLE_NAME}` SET `can_read`=1, `was_scanned`=1, 
                        `scan_attempts`= (SELECT `scan_attempts` FROM `{self.TABLE_NAME}`
                        WHERE `file_location`='{file}')+1
                        WHERE `file_location`='{file}';"""

        self._query_db(query, commit=True)

    def _conversion_successful(self, original_file, new_file):
        query = f"""UPDATE `{self.CONVERSION_TABLE}` SET `was_converted`=1, `new_location`='{new_file}', 
                    `new_extension`='{Path(new_file).suffix}', 
                    `convert_attempts`= (SELECT `convert_attempts` FROM `{self.CONVERSION_TABLE}`
                    WHERE `file_location`='{original_file}')+1
                    WHERE `file_location`='{original_file}';"""
        self._query_db(query, commit=True)

    def _conversion_failed(self, original_file):
        query = f"""UPDATE `{self.CONVERSION_TABLE}` SET `was_converted`=0, 
                `convert_attempts`= (SELECT `convert_attempts` FROM `{self.CONVERSION_TABLE}`
                WHERE `file_location`='{original_file}')+1
                WHERE `file_location`='{original_file}';"""
        self._query_db(query, commit=True)

    def _load_progress_from_db(self) -> dict[str, list[str]]:
        file_dict = {}
        self._push_converted_files()

        query = f"""SELECT `file_extension`, `file_location` FROM `{self.TABLE_NAME}` 
                    WHERE `scan_success`=0 AND `was_converted`=0 ORDER BY `scan_attempts` DESC;"""
        result = self._query_db(query)

        for row in result.fetchall():
            key, path = row
            if key in file_dict:
                file_dict[key].append(path)
            else:
                file_dict[key] = [path]

        return file_dict

    def _add_location_to_db(self, location):
        ext = Path(location).suffix

        query = f"""INSERT INTO `{self.TABLE_NAME}` (`file_location`,`file_extension`) 
            SELECT '{location}','{ext}'  FROM (select 1) 
            WHERE NOT EXISTS (SELECT * FROM `{self.TABLE_NAME}` 
                WHERE `file_location`='{location}' LIMIT 1)"""

        self._query_db(query, commit=True)

    def _copy_file(self, file, new_location):
        Path(new_location).mkdir(parents=True, exist_ok=True)
        shutil.copy(file, new_location)

    def is_valid_date(
        self,
        date: str,
        earliest_date=datetime.datetime(2020, 1, 1),
        latest_date=datetime.datetime(2024, 1, 1),
    ):
        try:
            val = parse(date)

            if val >= earliest_date and val <= latest_date:
                return True
            else:
                return False

        except ValueError:
            return False

    def convert_all_video_formats(self, extention, extention2):
        self._populate_conversion_table(extention, extention2)

        query = f"""SELECT `file_location` FROM `{self.CONVERSION_TABLE}` 
                WHERE `file_extension` = '{extention}' AND `was_converted` = 0 
                ORDER BY `convert_attempts` DESC;"""
        result = self._query_db(query)

        for row in result.fetchall():
            new_loc = self.starting_path + self.CONVERTED_FOLDER + "\\"
            new_file = new_loc + Path(row[0]).stem + extention2

            Path(new_loc).mkdir(parents=True, exist_ok=True)
            ret = subprocess.call(
                f'''ffmpeg -y -i "{row[0]}" "{new_file}"''', shell=True
            )
            if ret == 0:
                self._conversion_successful(row[0], new_file)
            else:
                self._conversion_failed(row[0])

    def scan_for_dates(
        self, file, frame_skip=0, scale_up=2, avi=False, giveup_frame=1500
    ):
        if avi:
            return

        capture = cv2.VideoCapture(file)
        if capture.isOpened() == False:
            return self._file_not_readable(file)

        frame_count = 0
        while capture.isOpened():
            print(f"Current frame: {frame_count}")

            ret, img = capture.read()
            if ret == 0 or frame_count >= giveup_frame:
                print("Could not read video")
                self._file_not_readable(file)
                break

            pytesseract.pytesseract.tesseract_cmd = (
                "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            )
            display = self.image_preprocess(img, scale_up=scale_up)

            # val = self.image_draw_boxes(display)
            # cv2.imshow("test", val)
            # cv2.waitKey(3)

            result = pytesseract.image_to_data(
                display,
                output_type=Output.DICT,
                config="-c tessedit_char_whitelist=0123456789/-",
            )
            matches = re.findall(
                "(([0-9]{1,2})(\/|-)([0-9]{1,2})(\/|-)([0-9]{4}|[0-9]{2}))",
                " ".join(result["text"]),
            )

            if len(matches) > 0:
                for dates in matches:
                    if self.is_valid_date(dates[0]):
                        print(f"successfully found valid date: {dates[0]}")
                        self._date_found(file, parse(dates[0]))

                        capture.release()
                        cv2.destroyAllWindows()
                        return

            frame_count += 1 + frame_skip
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        print("No date found!")
        self._no_date_found(file)

        capture.release()
        cv2.destroyAllWindows()  # destroy all opened windows

    # snippet for visualizing what tesseract "sees"
    def image_draw_boxes(self, image):
        results = pytesseract.image_to_data(image, output_type=Output.DICT, lang="eng")
        boxresults = pytesseract.image_to_boxes(
            image, output_type=Output.DICT, lang="eng"
        )

        for i in range(0, len(results["text"])):
            # extract the bounding box coordinates of the text region from the current result
            tmp_tl_x = results["left"][i]
            tmp_tl_y = results["top"][i]
            tmp_br_x = tmp_tl_x + results["width"][i]
            tmp_br_y = tmp_tl_y + results["height"][i]
            tmp_level = results["level"][i]
            conf = results["conf"][i]
            text = results["text"][i]

            if tmp_level == 5:
                cv2.putText(
                    image,
                    text,
                    (tmp_tl_x, tmp_tl_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                cv2.rectangle(
                    image, (tmp_tl_x, tmp_tl_y), (tmp_br_x, tmp_br_y), (0, 0, 255), 1
                )

            for j in range(0, len(boxresults["left"])):
                left = boxresults["left"][j]
                bottom = boxresults["bottom"][j]
                right = boxresults["right"][j]
                top = boxresults["top"][j]
                # cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 1)
        return image

    def image_preprocess(self, image, scale_up=2):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(
            img, None, fx=scale_up, fy=scale_up, interpolation=cv2.INTER_CUBIC
        )

        # TODO actually learn how to preprocess ....this was a waste of time.

        # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # kernel = np.ones((3, 3), np.uint8)
        # img = cv2.erode(img, kernel, iterations=1)
        # img = cv2.dilate(img, kernel, iterations=1)

        return img

    # type hinting to avoid confusion -- you're welcome future Damon. Return dict with ext as the key, loc as val.
    def find_all_file_types(
        self, starting_directory: str, extentions: tuple[str, ...]
    ) -> dict[str, list[str]]:
        for current_path, sub_dirs, files in os.walk(starting_directory):
            remove_dirs = [self.SORTED_FOLDER, self.GIVEUP_FOLDER]

            for dir in remove_dirs:
                if dir in sub_dirs:
                    sub_dirs.remove(f"{dir}")

            for file_name in files:
                if file_name.endswith(extentions):
                    key = Path(file_name).suffix.lower()

                    path = current_path + "\\" + file_name
                    self._add_location_to_db(path)

        return self._load_progress_from_db()


class Main:
    FILE_PATH = "C:\\USB 4\\"

    def __init__(self) -> None:
        print("Initializing program...")
        # start_location = os.path.abspath(os.path.dirname(sys.argv[0])) + "\\"
        start_location = self.FILE_PATH

        print(f"Starting location: {start_location}")

        ocr = AzpOCR(start_location)
        vids = ocr.find_all_file_types(start_location, (".mp4", ".wmv"))

        print("before convert")
        ocr.convert_all_video_formats(".avi", ".mp4")
        ocr.convert_all_video_formats(".m1v", ".mp4")

        for ext in list(vids.keys()):
            print(f"Video type: {ext}")

            # todo remove- bug in opencv breaks w/ avi files
            if ext == ".avi":
                continue

            for location in vids[ext]:
                print(f"Location: {location}\n")

                if ext == ".wmv":
                    ocr.scan_for_dates(
                        location, frame_skip=0, scale_up=4, giveup_frame=600
                    )
                else:
                    ocr.scan_for_dates(
                        location, frame_skip=3, scale_up=1, giveup_frame=800
                    )


if __name__ == "__main__":
    v = Main()
