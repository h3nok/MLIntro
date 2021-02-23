SELECT name FROM vinet.configurations

select * from vinet.get_frames('Hawk-Falcon', 20)

set role vinet_admin;


CREATE OR REPLACE FUNCTION vinet.select_frames_by_category_cursor(_category character varying, _limit integer)
 RETURNS refcursor
 LANGUAGE plpgsql
AS $function$
DECLARE
	ref refcursor;
BEGIN
	OPEN ref FOR
		SELECT * from vinet.get_frames(_category , _limit);
		
	RETURN ref;
END;
$function$; 

reset role;