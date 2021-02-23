from interop.executable import Executable
import os
import database_interface as dbi

TOOLS = r"E:\viNet_RnD\Tools"
DB = 'viNet'
HOST = '12.1.1.91'


def upload_model_to_db(pb_file, class_map, input_size=None, mean=127.5,
                       input_node='input',
                       output_node='output',
                       db=DB,
                       host=HOST, config=None, net_name=None) -> str:
    """
    Uploads viNet config to database

    @param mean:
    @param input_size:
    @param net_name:
    @param config:
    @param pb_file:
    @param class_map:
    @param input_node:
    @param output_node:
    @param db:
    @param host:
    @return:
    """

    exe_path = os.path.join(TOOLS, r"DatabaseTools\viNetConfigurationImporter.exe")
    exe = Executable(exe_path=exe_path)
    if not mean:
        mean = float(input("Enter mean values: "))

    assert mean
    if not input_size:
        input_size = int(input("Enter input size: "))
    assert input_size, "Must supply input size"
    if not config:
        config = str(input("Enter config name (press <enter> to use file name): "))

    if not config:
        config = os.path.basename(os.path.splitext(pb_file)[0])
    image_type = "BoundingBox"
    if not config:
        config = os.path.basename(pb_file).split('.')[0]

    if not net_name:
        net_name = config

    args = ["-b \"{}\"".format(pb_file),
            "-c \"{}\"".format(class_map),
            "-i {}".format(input_node),
            "-o {}".format(output_node),
            "-m {}".format(mean),
            "-s {}".format(input_size),
            "--configName {}".format(config),
            "--netName {}".format(net_name),
            "--imageType {}".format(image_type),
            "-d {}".format(db),
            "-h {}".format(host)
            ]

    exe.run(args, True)

    return config


def evaluate_candidate_formal(config=None, tag=None) -> bool:
    """
    Runs evaluation of a candidate model (config) on the supplied tag.
    This uses the C++ tool

    @param config:
    @param tag:

    @return:
    """
    assert config
    assert tag
    exe_path = os.path.join(TOOLS, "viNetClassifyTool\\viNetClassifyTool.exe")

    assert os.path.exists(exe_path)
    exe = Executable(exe_path)
    args = [f"-c {config}", f"-t \"{tag}\"", f"-h {HOST}", f"-d {DB}"]
    exe.run(args=args, block=True)

    return True


def get_dataset_tag_names(server):
    if not server:
        server = dbi.PgsqlInterface()
        server.connect()
    list_tags_query = "SELECT * FROM vinet.tags"

    server.execute(list_tags_query)

    tag = server.fetch_one()
    tags = list()
    counter = 0
    print("======================TAGS=================================")
    while tag:
        print("{}: {}".format(counter, tag[1]))
        tags.append(tag[1])
        counter += 1
        tag = server.fetch_one()

    return tags


def get_vinet_candidate_model_names(server) -> list:
    if not server:
        server = dbi.PgsqlInterface()
        server.connect()
    list_tags_query = "SELECT * FROM vinet.configurations"

    server.execute(list_tags_query)

    config = server.fetch_one()
    configs = list()
    counter = 0
    print("======================CONFIGS=================================")
    while config:
        print("{}: {}".format(counter, config[1]))
        configs.append(config[1])
        counter += 1
        config = server.fetch_one()

    return configs


def get_site_names(server) -> list:
    counter = 0
    if not server:
        server = dbi.PgsqlInterface()
        server.connect()
    query = "SELECT * from source.sites"
    server.execute(query)
    sites = list()
    site = server.fetch_one()

    while site:
        print("{}: {}".format(counter, site[1]))
        sites.append(site[1])
        site = server.fetch_one()
        counter += 1

    return sites


def get_config_id(server, config_name: str) -> str:
    get_site_id_query = "SELECT config_id FROM ijorquera.train_configs WHERE config_name = %(config_name)"
    server.execute_param(get_site_id_query, {"config_name": config_name})

    config_id = server.fetch_one()

    return config_id


def create_config_id(server, name: str, uuid: str):
    add_id_query = "INSERT INTO ijorquera.train_configs (config_id) VALUES (%(id)s)"
    server.commit_param(add_id_query, {"id": uuid})

    add_id_query = "INSERT INTO ijorquera.training_configurations (config_id) VALUES (%(id)s)"
    server.commit_param(add_id_query, {"id": uuid})

    add_name_q = "UPDATE ijorquera.train_configs SET config_name = %(name)s WHERE config_id = %(id)s"

    server.commit_param(add_name_q, {"name": name, "id": uuid})
