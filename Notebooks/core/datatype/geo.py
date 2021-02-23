from geopy import geocoders
import timezonefinder
import pytz


class GeographicalLocation(object):
    _timezone = None

    def __init__(self, lat=None, long=None, street=None, city=None, state=None, zipcode=None):
        """
        ADT representing geographical locations - either long, lat or a full address
        @param long:
        @param lat:
        @param street:
        @param city:
        @param state:
        @param zipcode:
        """
        assert (long and lat) or (street and city and state and zipcode), \
            "Must supply either longitude or latitude or a full address including zipcode"
        self.__latitude = lat
        self.__longitude = long
        self.__street = street
        self.__city = city
        self.__state = state
        self.__zipcode = zipcode

    @property
    def coordinates(self):
        return str.format("{},{}", self.latitude, self.longitude)

    @property
    def latitude(self):
        return self.__latitude

    @property
    def longitude(self):
        return self.__longitude

    @property
    def address(self):
        locator = geocoders.Nominatim(user_agent='google')
        location = locator.reverse(self.coordinates)

        return location.address

    def country_state_city(self):
        locator = geocoders.Nominatim(user_agent='google')
        location = locator.reverse(self.coordinates, exactly_one=True)
        address = location.raw['address']
        city = address.get_persisted_data('city', '')
        state = address.get_persisted_data('state', '')
        country = address.get_persisted_data('country', '')
        return city, state, country

    @property
    def timezone(self):
        if self._timezone is not None:
            return self._timezone
        else:
            tzf = timezonefinder.TimezoneFinder()
            # Get the tz-database-style time zone name (e.g. 'America/Vancouver') or None
            timezone_str = tzf.certain_timezone_at(lat=self.latitude, lng=self.longitude)

            timezone = pytz.timezone(timezone_str)
            self._timezone = timezone
            return self._timezone


class WindFarm(object):
    __name = None
    __geo_location = None

    def __init__(self, name, geo_location: GeographicalLocation):
        """
        ADT that represents a wind farm or a site
        @param name:
        @param geo_location:
        """
        self.__name = name
        self.__geo_location = geo_location

    @property
    def geo_location(self):
        return self.__geo_location

    @property
    def timezone(self):
        return self.__geo_location.timezone

    @property
    def coordinates(self):
        return self.__geo_location.coordinates

    @property
    def latitude(self):
        return self.__geo_location.latitude

    @property
    def longitude(self):
        return self.__geo_location.longitude

    @property
    def address(self):
        return self.__geo_location.address

    @property
    def name(self):
        return self.__name
