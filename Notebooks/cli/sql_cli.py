import database_interface as dbi

server = dbi.PgsqlInterface()
server.connect(name='frame_dist')

query = """
SELECT
	source.inspection_time as time 
FROM vinet.tagged_frames tag 
    JOIN viclassify.frame_results_latest t ON tag.frame_id =t.frame_id
	JOIN source.frame_data source ON source.frame_id = tag.frame_id
WHERE  
	tag.tag_id in (SELECT tag_id FROM vinet.tags WHERE name in ( 'Vattenfall V4 Validation'))
--ORDER BY source.inspection_time asc
group by source.inspection_time
"""

server.execute(query)
records = server.fetch_all()

print(len(records))

print(min(records))
print(max(records))
