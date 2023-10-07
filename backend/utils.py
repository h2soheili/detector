import datetime
import re
from typing import Optional

import pandas as pd


def date_to_number(date_input: datetime.datetime) -> int:
    if date_input is None:
        return None
    return int(str(date_input.year) + str(date_input.month).zfill(2) + str(date_input.day).zfill(2))


def is_null(input: str):
    result = True
    if input is not None and str(input).strip() != 'null' and str(input).strip() != 'undefined':
        result = False
    return result


def date_time_to_str(d: datetime):
    if isinstance(d, datetime.datetime):
        return d.__str__()


def to_date_time(str_representation) -> datetime.datetime:
    if isinstance(str_representation, pd.Timestamp):
        return str_representation.to_pydatetime(str_representation)
    if isinstance(str_representation, datetime.datetime):
        return str_representation
    result = str_representation
    if str_representation:
        if '+' in str_representation:
            str_representation = str_representation.split('+')[0]
        if '.' in str_representation:
            str_representation = str_representation[0: str_representation.index('.') + 7]

        if isinstance(str_representation, str) and len(str_representation) >= 8:
            date_format = "%Y-%m-%d %H:%M:%S"
            if 'T' in str_representation:
                date_format = '%Y-%m-%dT%H:%M:%S'
            if '.' in str_representation:
                date_format += ".%f"
            if str_representation.endswith('Z'):
                date_format += "Z"
            result = datetime.datetime.strptime(str_representation, date_format)
    return result


def str_fa_to_date_time(str_representation) -> datetime.datetime:
    result = None
    if str_representation and len(str_representation) >= 8:
        date_fa, time_fa = str_representation.split(' ')
        hour, minute, second = time_fa.split(':')
        hour, minute, second = int(hour), int(minute), int(second)
        persian = Persian(date_fa).gregorian_datetime()
        result = datetime.datetime(year=persian.year, month=persian.month, day=persian.day, hour=hour, minute=minute,
                                   second=second)
    return result


def str_to_time(str_time) -> datetime.time:
    time_parts = str_time.split(':')
    second_val = int(time_parts[2]) if len(time_parts) > 2 else 0
    return datetime.time(int(time_parts[0]), int(time_parts[1]), second_val)


def date_to_number(date_input: datetime.datetime) -> int:
    if date_input is None:
        return None
    return int(str(date_input.year) + str(date_input.month).zfill(2) + str(date_input.day).zfill(2))


def number_to_date(representation: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime.datetime:
    str_representation = str(representation)
    return datetime.datetime(int(str_representation[0:4]), int(str_representation[4:6]), int(str_representation[6:8]),
                             hour, minute, second)


def number_to_date_format(str_representation: str) -> datetime.datetime:
    if isinstance(str_representation, int):
        str_representation = str(str_representation)
    if str_representation is None:
        return None
    hour = 0
    minute = 0
    second = 0
    if len(str_representation) > 8:
        hour = int(str_representation[8:10])
        if len(str_representation) > 10:
            minute = int(str_representation[10:12])
            if len(str_representation) > 12:
                second = int(str_representation[12:14])
    date_converted = datetime.datetime(year=int(str_representation[0:4]), month=int(str_representation[4:6]),
                                       day=int(str_representation[6:8]), hour=hour, minute=minute, second=second)
    return date_converted


def str_to_date_time(str_representation, default_value=None) -> Optional[datetime.datetime]:
    result = default_value
    try:
        date_format = ''
        str_repr = str(str_representation).strip()
        if is_null(str_repr):
            return default_value

        date_part = str_repr
        if 'T' in str_repr:
            date_part = str_repr.split('T')[0]
            time_part = str_repr.split('T')[1]
        elif ' ' in str_repr:
            date_part = str_repr.split(' ')[0]
            time_part = str_repr.split(' ')[1]
        else:
            time_part = str_repr.replace(date_part, '')

        if date_part.count('-') >= 1:
            date_format += '%Y-%m'
            if date_part.count('-') >= 2:
                date_format += '-%d'
        else:
            if len(date_part) >= 4:
                date_format += '%Y'
            if len(date_part) >= 6:
                date_format += '%m'
            if len(date_part) >= 8:
                date_format += '%d'

        if time_part:
            if 'T' in str_repr:
                date_format += 'T'
            elif ' ' in str_repr:
                date_format += ' '

            if time_part.count(':') >= 1:
                date_format += '%H:%M'
                if time_part.count(':') >= 2:
                    date_format += ':%S'
            else:
                if len(time_part) >= 2:
                    date_format += '%H'
                if len(time_part) >= 4:
                    date_format += '%M'
                if len(time_part) >= 6:
                    date_format += '%S'
            if '.' in time_part:
                date_format += ".%f"
            if time_part.endswith('Z'):
                date_format += "Z"

        if date_format:
            result = datetime.datetime.strptime(str_repr, date_format)
    except:
        return default_value
    return result


class Persian:
    def __init__(self, *date):
        # Parse date
        if len(date) == 1:
            date = date[0]
            if type(date) is str:
                m = re.match(r'^(\d{4})\D(\d{1,2})\D(\d{1,2})$', date)
                if m:
                    [year, month, day] = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
                else:
                    raise Exception("Invalid Input String")
            elif type(date) is tuple:
                year, month, day = date
                year = int(year)
                month = int(month)
                day = int(day)
            else:
                raise Exception("Invalid Input Type")
        elif len(date) == 3:
            year = int(date[0])
            month = int(date[1])
            day = int(date[2])
        else:
            raise Exception("Invalid Input")

        # Check validity of date. TODO better check (leap years)
        if year < 1 or month < 1 or month > 12 or day < 1 or day > 31 or (month > 6 and day == 31):
            raise Exception("Incorrect Date")

        self.persian_year = year
        self.persian_month = month
        self.persian_day = day

        # Convert date
        d_4 = (year + 1) % 4
        if month < 7:
            doy_j = ((month - 1) * 31) + day
        else:
            doy_j = ((month - 7) * 30) + day + 186
        d_33 = int(((year - 55) % 132) * .0305)
        a = 287 if (d_33 != 3 and d_4 <= d_33) else 286
        if (d_33 == 1 or d_33 == 2) and (d_33 == d_4 or d_4 == 1):
            b = 78
        else:
            b = 80 if (d_33 == 3 and d_4 == 0) else 79
        if int((year - 19) / 63) == 20:
            a -= 1
            b += 1
        if doy_j <= a:
            gy = year + 621
            gd = doy_j + b
        else:
            gy = year + 622
            gd = doy_j - a
        for gm, v in enumerate([0, 31, 29 if (gy % 4 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]):
            if gd <= v:
                break
            gd -= v

        self.gregorian_year = gy
        self.gregorian_month = gm
        self.gregorian_day = gd

    def gregorian_tuple(self):
        return self.gregorian_year, self.gregorian_month, self.gregorian_day

    def gregorian_string(self, date_format="{}-{}-{}"):
        return date_format.format(self.gregorian_year, self.gregorian_month, self.gregorian_day)

    def gregorian_datetime(self):
        return datetime.date(self.gregorian_year, self.gregorian_month, self.gregorian_day)
