import database_interface as dbi

GET_TABELS = "SELECT table_name FROM information_schema.tables ORDER BY table_name"
GET_SCHEMAS = 'SELECT nspname FROM pg_catalog.pg_namespace'

FETCH_CLASSIFICATION_METRIC = """SELECT * FROM vinet.get_config_results_by_class('TOTW-Avangrid_v2.2 Live')"""

FETCH_TRUTH_ALL_PREDICTED = """SELECT 
	source.frame_id as FRAME_ID, 
	tc.name as TRUTH,
	cnc.name as OTHER_PRED,
	cnn.confidence as OTHER_PROB

FROM vinet.tagged_frames tag 
	JOIN viclassify.frame_results_latest t ON tag.frame_id =t.frame_id
	JOIN source.frame_data source ON source.frame_id = t.frame_id
 	JOIN source.classifications tc ON t.classification_id = tc.id
	JOIN vinet.frame_results_complete cnn ON cnn.frame_id = t.frame_id
	JOIN source.classifications cnc ON cnn.classification_id = cnc.id
WHERE
"""

FETCH_TRUTH_PREDICTED = """ SELECT 
	source.frame_id as FRAME_ID, 
	tc.name as TRUTH, 
	nc.name as PRED,
	nn.confidence as PROB

FROM vinet.tagged_frames tag 
	JOIN viclassify.frame_results_latest t ON tag.frame_id =t.frame_id
	JOIN source.frame_data source ON source.frame_id = t.frame_id
 	JOIN source.classifications tc ON t.classification_id = tc.id
 	JOIN vinet.frame_results_best nn ON nn.frame_id = t.frame_id 
 	JOIN source.classifications nc ON nn.classification_id =nc.id
WHERE  
"""

FETCH_TRUTH_PREDICTED_WITH_TIME = """ SELECT
	source.frame_id as FRAME_ID, 
	tc.name as TRUTH, 
	nc.name as PRED,
	nn.confidence as PROB,
	source.inspection_time as TIME

FROM vinet.tagged_frames tag 
	JOIN viclassify.frame_results_latest t ON tag.frame_id =t.frame_id
	JOIN source.frame_data source ON source.frame_id = t.frame_id
 	JOIN source.classifications tc ON t.classification_id = tc.id
 	JOIN vinet.frame_results_best nn ON nn.frame_id = t.frame_id 
 	JOIN source.classifications nc ON nn.classification_id =nc.id
WHERE  
"""

FETCH_TRUTH_PREDICTED_WITH_TIME_DISTINCT = """ SELECT DISTINCT ON (FRAME_ID)
	source.frame_id as FRAME_ID, 
	tc.name as TRUTH, 
	nc.name as PRED,
	nn.confidence as PROB,
	source.inspection_time as TIME

FROM vinet.tagged_frames tag 
	JOIN viclassify.frame_results_latest t ON tag.frame_id =t.frame_id
	JOIN source.frame_data source ON source.frame_id = t.frame_id
 	JOIN source.classifications tc ON t.classification_id = tc.id
 	JOIN vinet.frame_results_best nn ON nn.frame_id = t.frame_id 
 	JOIN source.classifications nc ON nn.classification_id =nc.id
WHERE  
"""

FETCH_BINARY_BB = """SELECT
        s.frame_id as FRAME_ID,
	    tc.name as TRUTH,
	    nc.name as VINET_STANDALONE_CLASS,
	    nn.confidence as VINET_CONFIDENCE, 
	    s.boundingbox_image frame_data
FROM
        vinet.frame_results_best nn
        JOIN viclassify.frame_results_latest t ON nn.frame_id = t.frame_id
        JOIN source.classifications nc ON nn.classification_id = nc.id
        JOIN source.classifications tc ON t.classification_id = tc.id
        JOIN source.frame_data s ON s.frame_id = nn.frame_id
WHERE
 	nn.config_id = '62812349-b296-4ff3-8d65-ac8d156f6b37'
	AND (tc.id = 3
	OR tc.id = 4
	OR tc.id = 8
	OR tc.id = 9
	OR tc.id = 10)
"""

FETCH_BINARY_BB_WITHOUT_RAVEN = """SELECT
        s.frame_id FRAME_ID,
	tc.name as TRUTH,
	nc.name as VINET_STANDALONE_CLASS,
	nn.confidence as VINET_CONFIDENCE,
	s.boundingbox_image frame_data
FROM
        vinet.frame_results_best nn
        JOIN viclassify.frame_results_latest t ON nn.frame_id = t.frame_id
        JOIN source.classifications nc ON nn.classification_id = nc.id
        JOIN source.classifications tc ON t.classification_id = tc.id
        JOIN source.frame_data s ON s.frame_id = nn.frame_id
WHERE
 	nn.config_id = '62812349-b296-4ff3-8d65-ac8d156f6b37'
	AND (tc.id = 3
	OR tc.id = 4
	OR tc.id = 8
	OR tc.id = 10)
"""

FETCH_BINARY_PADDED = """SELECT
        s.frame_id FRAME_ID,
	tc.name as TRUTH,
	nc.name as VINET_STANDALONE_CLASS,
	nn.confidence as VINET_CONFIDENCE,
	s.boundingbox_image frame_data
FROM
        vinet.frame_results_best nn
        JOIN viclassify.frame_results_latest t ON nn.frame_id = t.frame_id
        JOIN source.classifications nc ON nn.classification_id = nc.id
        JOIN source.classifications tc ON t.classification_id = tc.id
        JOIN source.frame_data s ON s.frame_id = nn.frame_id
WHERE
	nn.config_id = '5eb061d0-fc9e-471e-bc09-8fa61477ff31'
	AND (tc.id = 3
	OR tc.id = 4
	OR tc.id = 8
	OR tc.id = 9
	OR tc.id = 10)
"""
FETCH_TRAINING_FRAMES = """SELECT
        s.frame_id FRAME_ID,
        tc.name as TRUTH,
        nc.name as VINET_STANDALONE_CLASS,
        nn.confidence as VINET_CONFIDENCE,
        s.boundingbox_image frame_data
FROM
        vinet.frame_results_best nn
        JOIN viclassify.frame_results_latest t ON nn.frame_id = t.frame_id
        JOIN source.classifications nc ON nn.classification_id = nc.id
        JOIN source.classifications tc ON t.classification_id = tc.id
        JOIN source.frame_data s ON s.frame_id = nn.frame_id
WHERE
      	nn.config_id = '62812349-b296-4ff3-8d65-ac8d156f6b37'
        AND (tc.id = 3
        OR tc.id = 4
        OR tc.id = 8
        OR tc.id = 9
        OR tc.id = 10)
"""

FETCH_FRAMES_BY_ID = """SELECT
        s.frame_id FRAME_ID,
	tc.name as TRUTH,
	nc.name as VINET_STANDALONE_CLASS,
	nn.confidence as VINET_CONFIDENCE,
	s.boundingbox_image frame_data
FROM
        vinet.frame_results_best nn
        JOIN viclassify.frame_results_latest t ON nn.frame_id = t.frame_id
        JOIN source.classifications nc ON nn.classification_id = nc.id
        JOIN source.classifications tc ON t.classification_id = tc.id
        JOIN source.frame_data s ON s.frame_id = nn.frame_id
WHERE
 	nn.config_id = '62812349-b296-4ff3-8d65-ac8d156f6b37'
"""

REFRESH_CLASSIFICATION_COUNT = """REFRESH MATERIALIZED VIEW viclassify.classification_counts;"""

FETCH_LEGACY = """SELECT 
	source.frame_id as FRAME_ID, 
	sc.name as TRUTH, 
	nc.name as PRED,
	nn.confidence as PROB,
	source.boundingbox_image as frame_data
FROM vinet.tagged_frames tag 
	JOIN viclassify.frame_results_latest t ON tag.frame_id =t.frame_id
	JOIN source.frame_data source ON source.frame_id = t.frame_id
	JOIN source.classifications sc ON t.classification_id = sc.id
	JOIN vinet.frame_results_best nn ON nn.frame_id = t.frame_id 
	JOIN source.classifications nc ON nn.classification_id =nc.id
	WHERE tag_id = (SELECT tag_id from vinet.tags where name = 'TOTW_v2.4 Verification')
LIMIT 1000        
"""

FETCH_VEFIFICATION_SET_2_2_WITHOUT_RAVEN = """SELECT 
	source.frame_id as FRAME_ID, 
	tc.name as TRUTH, 
	nc.name as PRED,
	nn.confidence as PROB

FROM vinet.tagged_frames tag 
	JOIN viclassify.frame_results_latest t ON tag.frame_id =t.frame_id
	JOIN source.frame_data source ON source.frame_id = t.frame_id
 	JOIN source.classifications tc ON t.classification_id = tc.id
 	JOIN vinet.frame_results_best nn ON nn.frame_id = t.frame_id 
 	JOIN source.classifications nc ON nn.classification_id =nc.id
	
WHERE tag_id = (SELECT tag_id from vinet.tags where name = 'TOTW_v2.4 Verification')
	  AND nn.config_id = (SELECT config_id from vinet.configurations WHERE name='TOTW-Avangrid_v2.2 Live')
          AND (tc.id = 3
	        OR tc.id = 4
	        OR tc.id = 8
	        OR tc.id = 10)
"""
FETCH_VEFIFICATION_SET_2_2 = """SELECT 
	source.frame_id as FRAME_ID, 
	tc.name as TRUTH, 
	nc.name as PRED,
	nn.confidence as PROB

FROM vinet.tagged_frames tag 
	JOIN viclassify.frame_results_latest t ON tag.frame_id =t.frame_id
	JOIN source.frame_data source ON source.frame_id = t.frame_id
 	JOIN source.classifications tc ON t.classification_id = tc.id
 	JOIN vinet.frame_results_best nn ON nn.frame_id = t.frame_id 
 	JOIN source.classifications nc ON nn.classification_id =nc.id
	
WHERE tag_id = (SELECT tag_id from vinet.tags where name = 'TOTW_v2.4 Verification')
	  AND nn.config_id = (SELECT config_id from vinet.configurations WHERE name='TOTW-Avangrid_v2.2 Live')
"""

FETCH_WORKING_TRAIING_PADDED_FRAMES = """SELECT
        source.frame_id as FRAME_ID,
        tc.name as TRUTH,
        source.padded_image as FRAME_DATA

FROM vinet.tagged_frames tag
        JOIN viclassify.frame_results p3 ON tag.frame_id = p3.frame_id AND (p3.phase_id = 3 OR p3.phase_id=2 OR p3.phase_id = 4)
        JOIN source.frame_data source ON source.frame_id = p3.frame_id
        JOIN source.classifications tc ON p3.classification_id = tc.id

WHERE tag.tag_id IN 
       (SELECT tag_id FROM vinet.tags 
        WHERE name IN ('TOTW_v2.4 Training Working'))				   
"""

FETCH_E3_TRAINING_PRELIM = """SELECT
        source.frame_id as FRAME_ID,
        tc.name as TRUTH,
        source.padded_image as FRAME_DATA
FROM vinet.tagged_frames tag
        JOIN viclassify.frame_results p3 ON tag.frame_id = p3.frame_id AND (p3.phase_id = 2 OR p3.phase_id = 5)
        JOIN source.frame_data source ON source.frame_id = p3.frame_id
        JOIN source.classifications tc ON p3.classification_id = tc.id

WHERE tag.tag_id IN 
       (SELECT tag_id FROM vinet.tags 
        WHERE name IN ('E3_v2.2 Training'))
"""

FETCH_E3_22 = """
SELECT
        source.frame_id as FRAME_ID,
        tc.name as TRUTH,
        source.padded_image as FRAME_DATA
    FROM vinet.tagged_frames tag
    JOIN source.frame_data source
                ON source.frame_id = tag.frame_id
    JOIN source.classifications tc
                ON tag.truth_classification_id = tc.id
    WHERE tag.tag_id IN ( SELECT tag_id FROM vinet.tags 
    WHERE name IN ('E3_v2.2 Training'))
"""
FETCH_WORKING_AVANGRID_PADDED = """SELECT
        source.frame_id as FRAME_ID,
        tc.name as TRUTH,
        source.padded_image as FRAME_DATA

FROM vinet.tagged_frames tag
        JOIN viclassify.frame_results p3 ON tag.frame_id = p3.frame_id AND (p3.phase_id = 3 OR p3.phase_id=2 OR p3.phase_id = 4)
        JOIN source.frame_data source ON source.frame_id = p3.frame_id
        JOIN source.classifications tc ON p3.classification_id = tc.id

WHERE tag.tag_id IN 
       (SELECT tag_id FROM vinet.tags 
        WHERE name IN ('Avangrid_v2.4 Training Working'))
"""

FETCH_E3_BY_TAG_AND_CONFIG = """
WITH track_results_sum AS
(
	SELECT
		frame_truth.track_id,
		frame_results.classification_id,
		COUNT(*) as count,
		MAX(frame_results.confidence) as highest_conf
		
	FROM vinet.frame_results_best frame_results
	JOIN viclassify.frame_results_latest frame_truth
		ON frame_truth.frame_id = frame_results.frame_id
	JOIN vinet.tagged_frames tag
		ON tag.frame_id = frame_results.frame_id 
	WHERE
		frame_results.config_id = (SELECT config_id FROM vinet.configurations WHERE name = '{}')
		AND tag.tag_id = (SELECT tag_id FROM vinet.tags WHERE name = '{}')
	GROUP BY frame_truth.track_id, frame_results.classification_id
),
track_results AS (
	SELECT
		net_results.track_id,
		net_results.classification_id as net,
		truth_results.classification_id as truth
	FROM (
		SELECT 
			DISTINCT ON (track_id)
			track_id, classification_id
		FROM track_results_sum
		ORDER BY track_id, count DESC, highest_conf DESC
	) net_results
	JOIN viclassify.merged_track_results truth_results ON truth_results.track_id = net_results.track_id
),
counts AS (
	SELECT
		net, truth, COUNT(*)
	FROM track_results
	GROUP BY net, truth
)
SELECT 
	nn.name as nn,
	truth.name as truth,
	c.count
FROM counts c
JOIN source.classifications nn ON nn.id = c.net
JOIN source.classifications truth ON truth.id = c.truth
"""

FETCH_E3_TRACKS = """
WITH track_results_sum AS
(
	SELECT
		frame_truth.track_id,
		frame_results.classification_id,
		COUNT(*) as count,
		MAX(frame_results.confidence) as highest_conf
		
	FROM vinet.frame_results_best frame_results
	JOIN viclassify.frame_results_latest frame_truth
		ON frame_truth.frame_id = frame_results.frame_id
	WHERE
		frame_results.config_id = (SELECT config_id FROM vinet.configurations WHERE name = '{}')
	GROUP BY frame_truth.track_id, frame_results.classification_id
),
track_results AS (
	SELECT
		net_results.track_id,
		net_results.classification_id as net,
		truth_results.classification_id as truth
	FROM (
		SELECT 
			DISTINCT ON (track_id)
			track_id, classification_id
		FROM track_results_sum
		ORDER BY track_id, count DESC, highest_conf DESC
	) net_results
	JOIN viclassify.merged_track_results truth_results ON truth_results.track_id = net_results.track_id
),
counts AS (
	SELECT
		net, truth, COUNT(*)
	FROM track_results
	GROUP BY net, truth
)
SELECT 
	nn.name as nn,
	truth.name as truth,
	c.count
FROM counts c
JOIN source.classifications nn ON nn.id = c.net
JOIN source.classifications truth ON truth.id = c.truth
"""

FETCH_SINGLE = """
SELECT
    source.frame_id as FRAME_ID,
    tc.name as TRUTH,
    source.padded_image as FRAME_DATA
        FROM vinet.tagged_frames tag
            JOIN source.frame_data source
                ON source.frame_id = tag.frame_id
            JOIN source.classifications tc
                ON tag.truth_classification_id = tc.id
        WHERE tag.tag_id IN ( SELECT tag_id FROM vinet.tags 
        WHERE name IN ('E3_v2.2 Training'))
 	AND source.frame_id ='{}'
"""

FETCH_TRAINING_SET_INFO = """SELECT 
	frame_id, local_track_id, inspection_time, site_id 
    FROM source.frame_data
    WHERE frame_id IN (SELECT frame_id FROM vinet.tagged_frames WHERE tag_id = 
		(SELECT tag_id from vinet.tags WHERE name = '{}'))
"""


def APPEND(query, text):
    assert text is not None, "Must supply a valid text to append to query"
    return "{} \n\t{}".format(query, text)


def PREPEND(query, text):
    assert text is not None, "Must supply a valid text to append to query"
    return "{} \n\t{}".format(text, query)

