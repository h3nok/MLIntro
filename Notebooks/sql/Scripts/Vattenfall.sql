    select
--        tf.frame_id,
--        grp_class.name,
--        fd.boundingbox_image as frame_data
--	count(*)
	fd.inspection_time_local as time


    from "source".frame_data fd
--        join "source".frame_data fd on fd.frame_id = tf.frame_id
        join viclassify.frame_results_latest frl on frl.frame_id = fd.frame_id
        joing vinet.tagged_frames tf on tf.frame_id  = tf.frame_id 
--        join vinet.network_classification_group_entry grp ON grp.truth_classification_id = frl.classification_id
--        join source.classifications grp_class ON grp.group_classification_id = grp_class.id
        where tf.tag_id= (select tag_id from vinet.tags where name='Vattenfall V3 Training')
--    and grp.classification_group_id = (SELECT classification_group_id FROM vinet.network_classification_group
--    WHERE name = 'Vattenfall Proper v3')
	 and fd.site_id  in (select site_id from "source".sites where name in ('Vattenfall-Gotland')) and 
--    frl.review_time > '2020-9-27'  
    order by fd.inspection_time_local Asc
