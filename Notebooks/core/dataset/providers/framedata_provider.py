from providers.provider_base import DataProviderBase
from common.db.stored_procedures import StoredProcedures as sp
from frame_datum import FrameDatum
import pandas as pd


class PixelFrameProvider(DataProviderBase):
    __query__ = None
    __sproc__ = ""
    __server = None

    def __init__(self, server=None):
        """
        A database interface to fetch images (frame_data)
        @param server:
        """
        self.__server = server
        if not self.__server:
            self.__server = super().database_hook()
        self.__server.connect()
        self.__connection = self.__server.database_hook
        self._frame_data = pd.DataFrame()

    def __enter__(self, server=None):
        return self

    def __name__(self):
        return self.__class__.__name__

    def _validate(self):
        pass

    def _get(self, category, limit):
        assert self.__server
        try:
            if limit == 1:
                self._frame_data = sp.get_frames(category, limit, self.__connection)
                return FrameDatum(self._frame_data)

            else:
                assert isinstance(category, list)
                self._frame_data = pd.DataFrame()
                for cat in category:
                    if self._frame_data.empty:
                        self._frame_data = sp.get_frames(cat, limit, self.__connection)
                    else:
                        frames = sp.get_frames(cat, limit, self.__connection)
                        self._frame_data = pd.concat([self._frame_data,  frames])
                frames = []
                for index, row in self._frame_data.iterrows():
                    frame_id = row['frame_id']
                    frame_data = row['frame_data']
                    groundtruth = row['category']

                    frames.append(FrameDatum(frame_data=frame_data,
                                             frame_id=frame_id,
                                             groundtruth=groundtruth))

                return frames

        except (Exception, BaseException) as e:
            print(e)

    def fetch_one(self, category='Raven'):
        return self._get(category, 1)

    def fetch_many(self, categories=None, limit=10):
        return self._get(categories, limit)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._frame_data
        del self.__connection
        self.__server.close()


