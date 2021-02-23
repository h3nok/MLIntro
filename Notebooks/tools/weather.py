from datetime import datetime
import datetime
import math
import requests
import pandas as pd
import csv


class Weather:
    """
    Copied old code from vinet
    Please use local time for all methods except get_sunrise_sunet() which uses UTC time.
    Here are some helpful resources if you are trying to change any of the daily weather stuff or use the NCDC api
    overview: https://www.ncdc.noaa.gov/cdo-web/datasets
    Example API calls: https://www.ncdc.noaa.gov/cdo-web/webservices/v2#gettingStarted
    """
    __token = 'gFEmQrbbuXNeYCQmRvkxRpmTZNIhvlTJ'  # only used for daily data from NOAA
    __base_url_data = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'  # for daily data
    _latitude = 0.0  # North positive. South negative
    _longitude = 0.0  # East Positive. West Negative
    _zipcode = None  # for daily data

    _ncdc_station_id = None
    _allow_hourly_metrics = True  # if hourly is not available this will be false to speed up data collection
    # A lot of this is used to optimize accessing the csv file by caching each day
    _most_recent_date = None
    _most_recent_date_d = None
    _most_recent_daily_sum = None
    _most_recent_daily_sum_by_day_by_weather = None
    _most_recent_by_weather_date = None

    _station_data_path = None
    _year_of_hourly_data = None
    _day_data = None

    def __init__(self, lat, lon, zipcode=None):
        self._latitude = lat
        self._longitude = lon
        self._zipcode = zipcode

    @staticmethod
    def jd_to_date(jd):
        """
        Convert Julian Day to date.
        https://gist.github.com/jiffyclub/1294443
        Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
            4th ed., Duffet-Smith and Zwart, 2011.
        @param jd: Julian Day as float, see example
        :return:
        year : int
            Year as integer. Years preceding 1 A.D. should be 0 or negative.
            The year before 1 A.D. is 0, 10 B.C. is year -9.
        month : int
            Month as integer, Jan = 1, Feb. = 2, etc.
        day : float
            Day, may contain fractional part.

        Examples
        --------
        Convert Julian Day 2446113.75 to year, month, and day.

        jd_to_date(2446113.75)
        (1985, 2, 17.25)

        """
        jd = jd + 0.5

        F, I = math.modf(jd)
        I = int(I)
        A = math.trunc((I - 1867216.25) / 36524.25)

        if I > 2299160:
            B = I + 1 + A - math.trunc(A / 4.)
        else:
            B = I

        C = B + 1524
        D = math.trunc((C - 122.1) / 365.25)
        E = math.trunc(365.25 * D)
        G = math.trunc((C - E) / 30.6001)
        day = C - E + F - math.trunc(30.6001 * G)

        if G < 13.5:
            month = G - 1
        else:
            month = G - 13

        if month > 2.5:
            year = D - 4716
        else:
            year = D - 4715

        return year, month, day

    @staticmethod
    def days_to_hmsm(days):
        """
        https://gist.github.com/jiffyclub/1294443
        Convert fractional days to hours, minutes, seconds, and microseconds.
        Precision beyond microseconds is rounded to the nearest microsecond.
        @param days: float, A fractional number of days. Must be less than 1.
        :return:
        hour : int
            Hour number.
        min : int
            Minute number.
        sec : int
            Second number.
        micro : int
            Microsecond number.
        Examples
        days_to_hmsm(0.1)
        (2, 24, 0, 0)

        """
        hours = days * 24.
        hours, hour = math.modf(hours)

        mins = hours * 60.
        mins, min = math.modf(mins)

        secs = mins * 60.
        secs, sec = math.modf(secs)

        micro = round(secs * 1.e6)

        return int(hour), int(min), int(sec), int(micro)

    def jd_to_datetime(self, jd):
        """
        https://gist.github.com/jiffyclub/1294443
        Convert a Julian Day to a datetime object.
        @param jd: Julian day as float
        :return: datetime

        Examples
        --------
        #>>> jd_to_datetime(2446113.75)
        datetime(1985, 2, 17, 6, 0)

        """
        year, month, day = self.jd_to_date(jd)
        frac_days, day = math.modf(day)
        day = int(day)
        hour, min, sec, micro = self.days_to_hmsm(frac_days)

        return datetime.datetime(year, month, day, hour, min, sec, micro, datetime.timezone.utc)

    @staticmethod
    def date_to_jd(year, month, day):
        """
        https://gist.github.com/jiffyclub/1294443
        Convert a date to Julian Day.
        Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
            4th ed., Duffet-Smith and Zwart, 2011.
        @param year: Year as integer. Years preceding 1 A.D. should be 0 or negative.
        The year before 1 A.D. is 0, 10 B.C. is year -9.
        @param month: Month as integer, Jan = 1, Feb. = 2, etc.
        @param day: Day as float, may contain fractional part.
        :return: Julian Day as float

        Examples
        --------
        Convert 6 a.m., February 17, 1985 to Julian Day

        date_to_jd(1985,2,17.25)
        2446113.75
        """
        if month == 1 or month == 2:
            yearp = year - 1
            monthp = month + 12
        else:
            yearp = year
            monthp = month

        # this checks where we are in relation to October 15, 1582, the beginning
        # of the Gregorian calendar.
        if ((year < 1582) or
                (year == 1582 and month < 10) or
                (year == 1582 and month == 10 and day < 15)):
            # before start of Gregorian calendar
            B = 0
        else:
            # after start of Gregorian calendar
            A = math.trunc(yearp / 100.)
            B = 2 - A + math.trunc(A / 4.)

        if yearp < 0:
            C = math.trunc((365.25 * yearp) - 0.75)
        else:
            C = math.trunc(365.25 * yearp)

        D = math.trunc(30.6001 * (monthp + 1))
        jd = B + C + D + day + 1720994.5

        return jd

    def get_sunrise_sunset(self, date: datetime, time_zone=0):
        """
        Follows the equations from https://en.wikipedia.org/wiki/Sunrise_equation
        From a given date this will return the sunrise and sunset times
        sunset and sunrise time were validated using the noaa resource https://www.esrl.noaa.gov/gmd/grad/solcalc/

        provide all dates in UTC time and leave timezone as 0.
        If timezone is non utc time the method caller must deal with timezones and possible daylight savings
        :param date: desired date
        :param time_zone: should be set_ to 0 always. it gets very complicated if it isnt.
        Use the datetime method .astimezone(timezone.utc) when calling this method
        :return:  datetime variable for sunrise and then a datetime variable for sunset.
        """
        # region Constants
        # J*
        jd_n = self.date_to_jd(date.year, date.month, date.day) - self.date_to_jd(date.year, 1, 1)
        j_star = jd_n - (self._longitude / 360)
        # solar mean anomaly (M)
        m = (357.5291 + 0.98560028*j_star) % 360
        # Equation of the Center (C)
        c = (1.9148 * math.sin(math.radians(m))) + (.0200 * math.sin(math.radians(2 * m))) + (
                .0003*math.sin(math.radians(3 * m)))
        # Ecliptic Longitude
        ecl_lon = (m + c + 180 + 102.9372) % 360
        # Solar Transit (j_transit)
        # equation of time
        equation_of_time = (0.0053 * math.sin(math.radians(m))) - (.0069 * math.sin(math.radians(2 * ecl_lon)))
        j_transit = self.date_to_jd(date.year, 1, 1) + .5 + j_star + equation_of_time
        j_transit += (self.date_to_jd(2000, 1, 2) - self.date_to_jd(2000, 1, 2-(time_zone/24)))
        # solar declination
        sin_declination = math.sin(math.radians(ecl_lon)) * math.sin(math.radians(23.44))
        solar_declination = math.degrees(math.asin(sin_declination))
        # endregion
        cos_hour_angle_top = math.sin(math.radians(-0.83)) - (
                math.sin(math.radians(self._latitude)) * math.sin(math.radians(solar_declination)))
        cos_hour_angle_bottom = math.cos(math.radians(self._latitude)) * math.cos(math.radians(solar_declination))
        cos_hour_angle = cos_hour_angle_top / cos_hour_angle_bottom
        hour_angle = math.degrees(math.acos(cos_hour_angle))

        j_rise = self.jd_to_datetime(j_transit - (hour_angle / 360))
        j_set = self.jd_to_datetime(j_transit + (hour_angle / 360))

        return j_rise, j_set

    def get_season(self, date: datetime):
        """
        Pretty simple. uses provided date and latitude to determine the season based on the month
        :param date: desired date. time not required
        :return: season as a string
        """
        if date.month in [6, 7, 8]:
            if self._latitude >= 0:
                return 'Summer'
            else:
                return 'Winter'
        elif date.month in [9, 10, 11]:
            if self._latitude >= 0:
                return 'Fall'
            else:
                return 'Spring'
        elif date.month in [12, 1, 2]:
            if self._latitude >= 0:
                return 'Winter'
            else:
                return 'Summer'
        elif date.month in [3, 4, 5]:
            if self._latitude >= 0:
                return 'Spring'
            else:
                return 'Fall'

    def _find_closest_ncdc_station_id(self, desired_date):
        """
        ncdc is used for daily weather
        """
        # might add a date to this too
        ncdc_station_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations"
        params_location = f'limit=1000&extent={self._latitude - .5},{self._longitude - .5},{self._latitude + .5},{self._latitude + .5}'
        token = {'token': self.__token}
        r = requests.get(ncdc_station_url, params=params_location, headers=token)

        if r.status_code != 200:
            return None

        stations = r.json()['results']

        lowest_distance = None
        lowest_dist_station = None

        for s in stations:
            dt_mindate = datetime.datetime.strptime(s['mindate'], '%Y-%m-%d')
            dt_maxdate = datetime.datetime.strptime(s['maxdate'], '%Y-%m-%d')
            if not (dt_mindate < desired_date < dt_maxdate):
                continue
            dist_sqrd = math.pow(self._longitude - float(s['longitude']), 2) + math.pow(
                self._latitude - float(s['latitude']), 2)
            if lowest_distance is None or dist_sqrd < lowest_distance:
                lowest_distance = dist_sqrd
                lowest_dist_station = s

        if lowest_dist_station is None:
            return None

        return lowest_dist_station['id']

    def get_weather_daily(self, date: datetime, weather_type='PRCP'):
        """
        This method Assume the i + 1 th call is after the ith call. In other words this assumes that the data is ordered
        by time. for consecutive calls to this function
        Uses a Noaa Database to get_as_tfexample daily weather reports. requires API token, max 20,000 requests a day.
        this is optimized to use about 6-15 requests.
        :param date: date to get_as_tfexample daily report
        :param weather_type: 'PRCP, SNWD, TMAX, TMIN, TAVG. not all are available for all locations
        :return: returns np.array of all values. take average to get_as_tfexample 1 value
        """
        if self._ncdc_station_id is None:
            self._ncdc_station_id = self._find_closest_ncdc_station_id(date)
        if self._ncdc_station_id is None:
            return None
        startdate = date.strftime("%Y-%m-%d")  # not sure how to incorporate timezone here
        by_weather_set = False
        if self._most_recent_by_weather_date == startdate:
            by_weather = self._most_recent_daily_sum_by_day_by_weather
            by_weather_set = True
        elif self._most_recent_date_d is None or (date - self._most_recent_date_d) > datetime.timedelta(days=19) \
                or date < self._most_recent_date_d:
            end_dt = date + datetime.timedelta(days=20)
            enddate = end_dt.strftime("%Y-%m-%d")  # not sure how to incorporate timezone here

            token = {'token': self.__token}
            # passing as string instead of dict because NOAA API does not like percent encoding
            params_location = f'datasetid=GHCND&stationid={self._ncdc_station_id}&units=standard&startdate={startdate}&enddate={enddate}&limit=500'

            r = requests.get(self.__base_url_data, params=params_location, headers=token)

            try:
                # results comes in json form. Convert to dataframe
                df = pd.DataFrame.from_dict(r.json()['results'])
                if len(df['station'].unique()) < 1:
                    print('No stations')
                    return None
                self._most_recent_daily_sum = df[['datatype', 'date', 'value']]
                self._most_recent_date_d = date
                # Catch all exceptions for a bad request or missing data
            except:
                print("Error converting weather data to dataframe. Missing data?")
                return None

        if not by_weather_set:
            by_day = self._most_recent_daily_sum[self._most_recent_daily_sum['date'] == startdate+'T00:00:00']
            by_weather = by_day[by_day['datatype'] == weather_type]
            self._most_recent_by_weather_date = startdate
            self._most_recent_daily_sum_by_day_by_weather = by_weather
        return by_weather[['value']].to_numpy()

    def get_weather_csv_location(self):
        """
        This function finds the closests weather station that has the local climatolofical dataset
        http://jsonviewer.stack.hu/#http://www.ncei.noaa.gov/access/services/search/v1/data?dataset=local-climatological-data&startDate=2019-01-01&endDate=2019-01-03&boundingBox=-95.47371,53.89779,-117.47371,31.89779
        this link shows the result of the call down below.
        takes about a second or two
        :return:
        returns distance squared, and sets class variable self.station_data_path as the desired csv path
        distance may be None
        """
        date = datetime.datetime(2019, 1, 1)
        startdate = date.strftime("%Y-%m-%d")  # not sure how to incorporate timezone here
        end_dt = date + datetime.timedelta(days=2)
        enddate = end_dt.strftime("%Y-%m-%d")  # not sure how to incorporate timezone here

        # passing as string instead of dict because NOAA API does not like percent encoding
        base = 'https://www.ncei.noaa.gov/access/services/search/v1/data'
        dataset = 'local-climatological-data'
        call_params = 'dataset={}&startDate={}&endDate={}&boundingBox={},{},{},{}&limit=100' \
            .format(dataset, startdate, enddate, self._longitude + 11,
                    self._latitude + 11, self._longitude - 11, self._latitude - 11
                    )
        r = requests.get(base, params=call_params)

        try:
            j = r.json()['results']
            lowest_dis = None
            station_json = None
            for result in j:
                dif_sqrd = math.pow(self._longitude - float(result['location']['coordinates'][0]), 2) + math.pow(
                    self._latitude - float(result['location']['coordinates'][1]), 2)
                if lowest_dis is None or dif_sqrd < lowest_dis:
                    lowest_dis = dif_sqrd
                    station_json = result
            self._station_data_path = station_json['filePath']
            return lowest_dis
        # Catch all exceptions for a bad request or missing data
        except:
            print("Error converting weather data to dataframe. Missing data?")
            return None

    def get_weather_data(self, date, return_columns, as_np=False):
        """
        Like the daily weather this method assumes ordered data by time. Ex: If the ith data point has the time stamp
        2018-5-23T12:32:23 then all data points following should be after this one
        Gets the closest in time hourly report to the date provided. Some brief documentation of this API can be found
        here: https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
        Speed: 400 images/s. varies a decent amount. this slowness is due to pandas and its terrible optimization
        :param date: datetime to get_as_tfexample sample. provide hours and minutes
        :param return_columns: Can be one of or multiple of the following:
                    'HourlyPrecipitation', 'HourlySkyConditions', 'HourlyRelativeHumidity',
                     'HourlyVisibility', 'HourlyWindSpeed', 'HourlyDryBulbTemperature'
        :param as_np:
        Returns np.array vs pandas.DataFrame
        :return:
        when as_np is true returns np.array with values in order of return_columns then DATE of samples. when false
        returns DataFrame with key-Values. Please not it is possible to receive None as the return. this is likely
        a result of lack of data. it is also possible to receive '' for a data point. Returns ['Too Far'] if the site is
        too far from the closest station
        """

        str_date = date.strftime("%Y-%m-%d")
        # Get CSV file location
        if self._station_data_path is None:
            dist = float(self.get_weather_csv_location())
            if dist > 5:
                print('Squared Distance from site is: {}. Not recommended'.format(dist))
                return ['Too Far']
            if self._station_data_path is None:
                return
        # Load CSV file and get_as_tfexample year data
        if self._most_recent_date is None or self._most_recent_date.year != date.year:
            self._most_recent_date = None
            split_path = self._station_data_path.split('/')
            split_path[-2] = str(date.year)
            self._station_data_path = '/'.join(split_path)
            data_call = "https://www.ncei.noaa.gov" + self._station_data_path
            r = requests.get(data_call)
            try:
                # results comes in json form. Convert to dataframe
                decoded_content = r.content.decode('utf-8')
                cr = csv.reader(decoded_content.splitlines(), delimiter=',')
                data_frame = pd.DataFrame(cr)
                new_header = data_frame.iloc[0]  # grab the first row for the header
                data_frame = data_frame[1:]  # take the data less the header row
                data_frame.columns = new_header  # set_value the header row as the df header
                self._year_of_hourly_data = data_frame[
                    ['DATE', 'HourlyPrecipitation', 'HourlySkyConditions', 'HourlyRelativeHumidity',
                     'HourlyVisibility', 'HourlyWindSpeed', 'HourlyDryBulbTemperature']]

            except:
                print("Error converting weather data to dataframe. Missing data?")
                return None
        if self._most_recent_date is None or (
                self._most_recent_date.month != date.month or self._most_recent_date.day != date.day):
            # Get Data for specified day
            self._day_data = self._year_of_hourly_data[
                self._year_of_hourly_data['DATE'].str.contains(str_date, regex=False)]
            # Delete all data until that day. this assumes the date being provided is ordered
            if len(self._day_data.index) < 1:
                print(date, 'day_data empty')
                return None
            to_keep = int(self._day_data.index[-1]) - int(self._year_of_hourly_data.index[0]) + 1
            self._year_of_hourly_data = self._year_of_hourly_data[to_keep:]
            self._year_of_hourly_data.reindex()
            self._day_data.reindex()

        self._most_recent_date = date
        return_columns.append('DATE')
        df = self._day_data[return_columns]

        # I have tried a lot of different methods, this one isnt recommended or perfect but its the fastest ive found
        for index, row in df.iterrows():
            time_stamp = datetime.datetime.strptime(row['DATE'], '%Y-%m-%dT%H:%M:%S')
            # This is a small work around to hopefully improve the speed
            if abs(time_stamp - date) < datetime.timedelta(minutes=61):
                if as_np:
                    return row.to_numpy()
                else:
                    return row
        print(date)
        return None

    def get_rain(self, date):
        """
        returns either hourly or daily rain statistic, or None if nothing is available
        """
        hourly_rain = None
        if self._allow_hourly_metrics:
            hourly_rain = self.get_hourly_rain(date)
        if hourly_rain is not None and len(hourly_rain) == 2:
            return hourly_rain[0]
        else:
            self._allow_hourly_metrics = False
            daily_rain = self.get_weather_daily(date, 'PRCP')
            if daily_rain is not None and len(daily_rain) > 0:
                return daily_rain[0][0]
            else:
                return None  # Might change to 0

    def get_hourly_rain(self, date, as_np=True):
        return self.get_weather_data(date, return_columns=['HourlyPrecipitation'], as_np=as_np)

    def get_hourly_visibility(self, date, as_np=True):
        return self.get_weather_data(date, return_columns=['HourlyVisibility'], as_np=as_np)

    def get_hourly_wind_speed(self, date, as_np=True):
        return self.get_weather_data(date, return_columns=['HourlyWindSpeed'], as_np=as_np)

    def get_hourly_humidity(self, date, as_np=True):
        return self.get_weather_data(date, return_columns=['HourlyRelativeHumidity'], as_np=as_np)

    def get_hourly_temperature(self, date, as_np=True):
        return self.get_weather_data(date, return_columns=['HourlyDryBulbTemperature'], as_np=as_np)

    def get_hourly_sky_conditions(self, date, as_np=True):
        return self.get_weather_data(date, return_columns=['HourlySkyConditions'], as_np=as_np)


if __name__ == '__main__':
    # 0: -106.47371, 42.89779, "CASPER NATRONA CO AIRPORT, WY US"
    lat_test = 42.89779
    lon_test = -106.47371
    zipcode_test = 82637

    w = Weather(lat=lat_test, lon=lon_test, zipcode=zipcode_test)
    year = 2019
    month = 8
    i = 4

    np_weather_val = w.get_weather_data(datetime.datetime(year, month, i, 12, 59), ['HourlyVisibility'], as_np=True)
    print(np_weather_val)
    np_weather_val = w.get_weather_data(datetime.datetime(year, month, i+1, 12, 59), ['HourlyVisibility'], as_np=True)
    print(np_weather_val)
