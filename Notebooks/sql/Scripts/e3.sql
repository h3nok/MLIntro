--E3, viNet v2.2 dataset 
            SELECT
                COUNT(*)
            FROM vinet.tagged_frames tag
                JOIN source.frame_data src ON tag.frame_id = src.frame_id
                JOIN vinet.network_classification_group_entry grp
       	            ON tag.truth_classification_id = grp.truth_classification_id
       	        JOIN source.classifications grp_class ON grp.group_classification_id = grp_class.id
                WHERE
       	            tag.tag_id = (SELECT tag_id FROM vinet.tags WHERE name = 'E3_v2.2 Training')
                AND
                    grp.classification_group_id = (
                    SELECT classification_group_id
      	            FROM vinet.network_classification_group
      	            WHERE name = 'E3 4 Class')
            
LIMIT 1000



select 
		tf.frame_id, 
		grp_class.name,
		fd.padded_image as frame_data
	from vinet.tagged_frames tf
		join "source".frame_data fd on fd.frame_id = tf.frame_id 
		join viclassify.frame_results_latest frl on frl.frame_id = tf.frame_id 
		join vinet.network_classification_group_entry grp ON grp.truth_classification_id = frl.classification_id 
		join source.classifications grp_class ON grp.group_classification_id = grp_class.id
	where tf.tag_id= (select tag_id from vinet.tags where name='Vattenfall Training V3') 
	        and	
	grp.classification_group_id = (SELECT classification_group_id FROM vinet.network_classification_group
		WHERE name = 'Vattenfall Proper v3')
order by tf.frame_id    	