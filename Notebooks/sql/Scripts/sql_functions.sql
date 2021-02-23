
CREATE OR REPLACE FUNCTION append_frames_to_tag(_tag_name text, _site_name text, _groundtruth text)
INSERT INTO vinet.tagged_frames (tag_id, frame_id, tagged_time, truth_classification_id, truth_row_id)
with
	tag as (select tag_id from vinet.tags where name = _tag_name),
	frame as (
		select
			fi.frame_id as frame_id,
			c.id as truth_classification_id,
			frl.frame_row_id as truth_row_id

		from "source".frame_info fi
			join viclassify.frame_results_latest frl on fi.frame_id = frl.frame_id
			join "source".classifications c on frl.classification_id = c.id
			where fi.site_id in (select site_id from source.sites
			where "name" = _site_name) and c."name"  = _groundtruth
			and frl.track_id  is not null
		limit 10
	)
	select tag.tag_id, frame.frame_id, Now(), frame.truth_classification_id, frame.truth_row_id
	from tag, frame;
