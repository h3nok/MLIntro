from datetime import datetime, timedelta
import os
from unittest import TestCase
from tools.weather import Weather

# 0: -106.47371, 42.89779, "CASPER NATRONA CO AIRPORT, WY US"
LAT = 42.89779
LON = -106.47371
ZIP = 82637


class TestWeather(TestCase):
    def test_jd_to_date(self):  # look at function for examples for the first few tests here
        # base case
        assert Weather.jd_to_date(0) == (-4713 + 1, 1, 1.5)  # start time for Julian time is year -4713 Jan 1st at hour 12, but we include 0 as a year
        assert Weather.jd_to_date(1721424) == (1, 1, 1.5)  # CE

        # on day is one value diff
        v = 2141425  # make random
        a = Weather.jd_to_date(v)
        a2 = Weather.jd_to_date(v+1)
        assert a == (1150, 11, 26.5)
        assert a[0] == a2[0]
        assert a[1] == a2[1]
        assert a[2]+1 == a2[2]

    def test_days_to_hmsm(self):
        assert Weather.days_to_hmsm(0) == (0, 0, 0, 0)

        num = 0.5
        a = Weather.days_to_hmsm(num)
        td = timedelta(hours=a[0], minutes=a[1], seconds=a[2], microseconds=a[3])
        a1 = Weather.days_to_hmsm(num + 1.0/24.0)  # one hours

        td1 = timedelta(hours=a1[0], minutes=a1[1], seconds=a1[2], microseconds=a1[3])
        assert td1.total_seconds() - td.total_seconds() == 60*60

        a2 = Weather.days_to_hmsm(num + (1.0/24.0)/60.0)  # one min
        td2 = timedelta(hours=a2[0], minutes=a2[1], seconds=a2[2], microseconds=a2[3])
        assert td2.total_seconds() - td.total_seconds() == 60

        # we could keep going

    def test_jd_to_datetime(self):
        assert False

    def test_date_to_jd(self):
        assert False

    def test_get_sunrise_sunset(self):
        # this isn't really a good way to test this I don't think.

        lat1 = 39.74
        lon1 = -104.99
        zip1 = 80202

        lat2 = 45.52
        lon2 = -122.65
        zip2 = -1

        lat3 = 25.77
        lon3 = -80.18
        zip3 = -1

        time1 = datetime(2020, 7, 10)
        time2 = datetime(2019, 2, 17)
        time3 = datetime(1980, 1, 1)

        w1 = Weather(lat1, lon1, zip1)
        w2 = Weather(lat2, lon2, zip2)
        w3 = Weather(lat3, lon3, zip3)

        res = [w1.get_sunrise_sunset(time1), w1.get_sunrise_sunset(time2), w1.get_sunrise_sunset(time3),
               w2.get_sunrise_sunset(time1), w2.get_sunrise_sunset(time2), #w2.get_sunrise_sunset(time3),
               w3.get_sunrise_sunset(time1), w3.get_sunrise_sunset(time2), w3.get_sunrise_sunset(time3)]

        val = [(datetime(2020, 7, 10, 11, 41), datetime(2020, 7, 11, 2, 29)),  # https://www.esrl.noaa.gov/gmd/grad/solcalc/
               (datetime(2019, 2, 17, 13, 50), datetime(2019, 2, 18, 0, 38)),
               (datetime(1980, 1, 1, 14, 21), datetime(1980, 1, 1, 23, 46)),
               (datetime(2020, 7, 10, 12, 32), datetime(2020, 7, 11, 3, 59)),
               (datetime(2019, 2, 17, 15, 10), datetime(2019, 2, 18, 1, 40)),
               #(datetime(1980, 1, 1, 17, 37), datetime(1980, 1, 2, 0, 42)),
               (datetime(2020, 7, 10, 10, 37), datetime(2020, 7, 11, 0, 15)),
               (datetime(2019, 2, 17, 11, 55), datetime(2019, 2, 17, 23, 15)),
               (datetime(1980, 1, 1, 12, 7), datetime(1980, 1, 1, 22, 41))]

        for i in range(len(res)):
            assert (datetime(res[i][0].year, res[i][0].month, res[i][0].day, res[i][0].hour,
                             int(round(res[i][0].minute + res[i][0].second / 60.0))),
                    datetime(res[i][1].year, res[i][1].month, res[i][1].day, res[i][1].hour,
                             int(round(res[i][1].minute + res[i][1].second / 60.0)))) == val[i]

    def test_get_season(self):
        lat1 = 39.74
        lon1 = -104.99
        zip1 = -1

        lat2 = 52.953508
        lon2 = 47.7733943
        zip2 = -1

        lat3 = -33.4724728
        lon3 = -70.910022
        zip3 = -1

        d1 = datetime(2019, 4, 12)
        d2 = datetime(2016, 12, 1)
        d3 = datetime(2020, 6, 20)
        d4 = datetime(2000, 9, 7)

        w1 = Weather(lat1, lon1, zip1)
        w2 = Weather(lat2, lon2, zip2)
        w3 = Weather(lat3, lon3, zip3)

        res = [
            w1.get_season(d1),
            w1.get_season(d2),
            w1.get_season(d3),
            w1.get_season(d4),

            w2.get_season(d1),
            w2.get_season(d2),
            w2.get_season(d3),
            w2.get_season(d4),

            w3.get_season(d1),
            w3.get_season(d2),
            w3.get_season(d3),
            w3.get_season(d4),
        ]

        val = ['Spring', 'Winter', 'Summer', 'Fall', 'Spring', 'Winter', 'Summer',
               'Fall', 'Fall', 'Summer', 'Winter', 'Spring']

        assert res == val

    def test_get_weather_daily(self):
        t1 = datetime(2019, 11, 9, 11, 3)

        from customer.customers import CustomerObjectMap
        cust = CustomerObjectMap["Vattenfall"]()

        w1 = Weather(cust.geo_location.latitude, cust.geo_location.longitude,)
        res1 = w1.get_weather_daily(t1, 'PRCP')
        print(res1)

    def test_get_weather_csv_location(self):
        assert False

    def test_get_weather_data(self):
        assert False

    def test_customer_locations_with_rain(self):
        t1 = datetime(2019, 11, 9)

        from customer.customers import CustomerObjectMap
        cust = CustomerObjectMap["Vattenfall"]()

        w1 = Weather(cust.geo_location.latitude, cust.geo_location.longitude)

        res1 = w1.get_hourly_rain(t1)

    def test_get_rain(self):
        dt = datetime(2019, 11, 9, 11, 45)

        from customer.customers import CustomerObjectMap
        cust = CustomerObjectMap["Vattenfall"]()

        weather = Weather(cust.geo_location.latitude, cust.geo_location.longitude)

        rain = weather.get_rain(dt)
        print(rain)

    def test_get_hourly_rain(self):

        t1 = datetime(2019, 11, 9)

        w = Weather(LAT, LON)  # ZIP
        res1 = w.get_hourly_rain(t1)

        assert len(res1) == 2
        assert t1.strftime("%Y-%m-%d") in res1[1]

        t2 = datetime(2020, 6, 9)

        w = Weather(LAT, LON, ZIP)
        res2 = w.get_hourly_rain(t2)

        assert len(res2) == 2
        assert t2.strftime("%Y-%m-%d") in res2[1]

    def test_get_hourly_visibility(self):  # this sounds useful but i don't think it actually is lol
        assert False

    def test_get_hourly_wind_speed(self):
        assert False

    def test_get_hourly_humidity(self):
        assert False

    def test_get_hourly_temperature(self):
        assert False

    def test_get_hourly_sky_conditions(self):
        assert False
