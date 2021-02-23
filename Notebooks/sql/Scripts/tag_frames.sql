-- First aggregate all the new frames we care about into frames and tracks
--  - Add an index_ column to the tracks that rank the tracks in order which
--      we want to add to the validation set (random for now)
----------------------------------------------------------------------------
DROP TABLE IF EXISTS _TAGGING_NewFrames;
CREATE TEMPORARY TABLE _TAGGING_NewFrames AS (
	select
	frl.track_id,
	fi.frame_id,
	grp_class."name" as truth_name,
	grp_class.id as truth_id,
	frl.frame_row_id 
	from "source".frame_info fi 
		join viclassify.frame_results_latest frl on fi.frame_id = frl.frame_id 
		join vinet.network_classification_group_entry ncge on frl.classification_id = ncge.truth_classification_id 
		join "source".classifications grp_class on ncge.group_classification_id = grp_class.id 
	where ncge.classification_group_id in (select classification_group_id 
	from vinet.network_classification_group ncg2 where ncg2."name" = '<classification group name goes here>')
	and fi.site_id in (select site_id from source.sites where "name" ='<site name>')
);

DROP TABLE IF EXISTS _TAGGING_NewTracks;
CREATE TEMPORARY TABLE _TAGGING_NewTracks AS (
	SELECT
		track.*,
		-- Rank the tracks just by track id so we get a random sample
		ROW_NUMBER() OVER (
			PARTITION BY track.truth_id
			ORDER BY track.track_id
		) AS index_
	FROM (
		SELECT 
			track_id,
			truth_name,
			truth_id,
			COUNT(*) as frames
		FROM _TAGGING_NewFrames
		GROUP BY
			track_id,
			truth_name,
			truth_id
		) track
);
	
-- Do the work.
--  - iterate over each distinct classification and get the amount of data we want to tag in validation
--  --> Find the appropriate tracks to tag as validation based on the frame/track counts
--  --> tag the frames in one of the two tags
DO $$
DECLARE
	current_class RECORD;
	current_validation_track_count int;
	current_validation_frame_count int;
	track_batch_size int;
	validation_tag_id uuid;
	training_tag_id uuid;
BEGIN
	validation_tag_id := '<validation tag id>';
	training_tag_id := '<training tag id>';
		
	FOR current_class IN 
		SELECT
			truth_id,
			truth_name,
			count(*) as tracks,
			sum(frames) as frames,
			count(*) * 0.050 as validation_target_tracks,
			count(*) * 0.100 as validation_max_tracks,
			sum(frames) * 0.20 as validation_target_frames
		FROM _TAGGING_NewTracks
		GROUP BY truth_id, truth_name
	LOOP
		
		RAISE NOTICE '%', current_class.truth_name;
		
		current_validation_frame_count := 0;
		current_validation_track_count := 0;
		
		-- set the minimum number of tracks to process
		track_batch_size := TRUNC(current_class.validation_target_tracks);
		
		-- Loop until we have enough frames or reach the maximum number of tracks
		-- once this is don
		WHILE
			current_validation_frame_count < current_class.validation_target_frames AND
			current_validation_track_count < current_class.validation_max_tracks
		LOOP
			SELECT
				COUNT(*) + current_validation_track_count,
				COALESCE( SUM(t.frames), 0 ) + current_validation_frame_count
			INTO
				current_validation_track_count,
				current_validation_frame_count
			FROM _TAGGING_NewTracks t
			WHERE
				t.truth_id = current_class.truth_id
		 		AND current_validation_track_count < t.index_ AND t.index_ <= current_validation_track_count + track_batch_size
			;
			-- now that we have the minimum tracks, just get 1 more at a time
			track_batch_size := 1;
		
			RAISE NOTICE '%, %, %, %', current_validation_track_count, current_validation_frame_count, current_class.validation_target_tracks, current_class.validation_target_frames;
		END LOOP;
		
		INSERT INTO vinet.tagged_frames (
			tag_id, frame_id, tagged_time, truth_classification_id, truth_row_id
		)
		SELECT
			CASE
				WHEN index_ <= current_validation_track_count THEN 
					validation_tag_id
				ELSE
					training_tag_id
			END,
			f.frame_id,
			NOW(),
			f.truth_id,
			f.frame_row_id
		FROM _TAGGING_NewFrames f
		JOIN _TAGGING_NewTracks t ON f.track_id = t.track_id
		WHERE f.truth_id = current_class.truth_id;
	END LOOP;
END;
$$
