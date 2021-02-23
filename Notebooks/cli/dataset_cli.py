import database_interface as dbi
from database.data_mappers.classification_group import  viNetClassGroup
from database.data_mappers.dataset_tag import DatasetTag, DatasetTagMapper
from database.data_mappers.vinet_site import viNetSite

server = dbi.PgsqlInterface()
# server.connect(name='frame_dist')


class SiteData:

    """
    Main interface to a site and all its data
    Here is where you prepare everything needed to get raw data converted to tfrecords
    TODO -
    """

    def __init__(self,
                 site: viNetSite,
                 train_tag: DatasetTag,
                 val_tag: DatasetTag,
                 class_group: viNetClassGroup):
        """

        @param site:
        @param train_tag:
        @param val_tag:
        @param class_group:
        """
        assert site
        assert train_tag
        assert val_t

        self._train_tag = train_tag
        self._val_tag = val_tag

        with DatasetTagMapper() as dtm:
            self._train_tag_id = dtm.insert(train_tag)
            self._val_tag_id = dtm.insert(val_tag)

        self._site = site
        self._train_tag.set_id(self._train_tag_id)
        self._val_tag.set_id(self._val_tag_id)
        self._class_group = class_group
        # assert self._site.has_data()

    def tfrecord_prep_info(self):
        info = f"""
        Site name: {self._site.name}
        Train tag: {self._train_tag_id} 
        Val tag: {self._val_tag_id}
        Class Group: {self._class_group.name}
        """

        return info


if __name__ == '__main__':
    s = viNetSite(name="Vattenfall-Gotland")
    train_t = DatasetTag(name='Vattenfall-Train-viAi-Test', comment='viAi unittest', tag_type='train')
    val_t = DatasetTag(name='Vattenfall-Val-viAi-Test', comment='viAi unittest', tag_type='val')
    cg = viNetClassGroup(name='Vattenfall Proper v4')
    ds = SiteData(s, train_tag=train_t, val_tag=val_t, class_group=cg)
    print(ds.tfrecord_prep_info())
