select
    tf.frame_id as frame_id,
    grp_class.name as groundtruth,
    fd.boundingbox_image as frame_data
from vinet.tagged_frames tf
    join "source".frame_data fd on fd.frame_id = tf.frame_id
    join viclassify.frame_results_latest frl on frl.frame_id = tf.frame_id
    join vinet.network_classification_group_entry grp ON grp.truth_classification_id = frl.classification_id
    join source.classifications grp_class ON grp.group_classification_id = grp_class.id
    where tf.tag_id= (select tag_id from vinet.tags where name='Goldwind-Cattle Hill Training v1 (WTE, Raven,Other-Avian, Hawk-Falcon)')
        and
    grp.classification_group_id = (SELECT classification_group_id FROM vinet.network_classification_group
        WHERE name = 'Goldwind Proper V2')
