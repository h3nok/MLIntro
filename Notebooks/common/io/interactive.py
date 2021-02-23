import sys

from common.db.stored_procedures import StoredProcedures as sp
from customer.customers import CustomerObjectMap
from common.io.database import get_site_names


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def query_tag(db_connection, customer):
    """
    Query the database for available tags
    @param db_connection:
    @param customer:
    @return:
    """
    if isinstance(customer, str):
        customer = CustomerObjectMap[customer]()
    tags, _ = sp.dataset_tags(db_connection)
    tag = None
    show_all = False
    while tag is None:
        for i in range(len(tags)):
            if show_all or customer.name.lower() in tags[i].lower():
                print(f"{i}. {tags[i]}")
        print(f"{len(tags)}. Show all/ Show less")
        print("\n")

        try:
            ind = int(input("Select a TAG from the list above (int): "))
            if ind == len(tags):
                show_all = not show_all
                continue
            tag = tags[ind]
        except (KeyError, IndexError, ValueError):
            print("Could not understand input. Insure input is of type int and in the range")
            tag = None
        print("\n")

    return tag


def query_config(connection, customer):
    """

    @param connection:
    @param customer:
    @return:
    """
    # Config
    configs = sp.vinet_configs(connection)
    config = None

    show_all = False
    if isinstance(customer, str):
        customer = CustomerObjectMap[customer]()
    while config is None:
        for i in range(len(configs)):
            if show_all or customer.name.lower() in configs[i].lower():
                print(f"{i}. {configs[i]}")
        print(f"{len(configs)}. Show all/ Show less")
        print("\n")

        try:
            ind = int(input("Select a CONFIG from the list above (int): "))
            if ind == len(configs):
                show_all = not show_all
                continue
            config = configs[ind]
        except (KeyError, IndexError, ValueError):
            print("Could not understand input. Insure input is of type int and in the range")
            config = None
        print("\n")

    return config


def query_customer():
    """

    @return:
    """
    customer = None

    while customer is None:
        customers = list(CustomerObjectMap.keys())
        for i in range(len(customers)):
            print(f"{i}. {customers[i]}")
        print("\n")

        try:
            customer = CustomerObjectMap[customers[int(input("Select a customer from the list above (int): "))]]()
        except (KeyError, IndexError, ValueError):
            print("Could not understand input. Insure input is of type int and in the range")
            customer = None
        print("\n")

    return customer


def query_site(server):
    sites = get_site_names(server)
    index = int(input("\nSelect index of site from above (int)>>: "))
    return sites[index]


def query_class_group(server):
    """

    @param server:
    @return:
    """
    classification_groups = list(sp.vinet_classification_groups(server.database_hook)["name"])
    c_group = None
    print("""\nCLASSIFICATION GROUP""")
    print("______________________")
    while c_group is None:
        for i in range(len(classification_groups)):
            print(f"{i}. {classification_groups[i]}")
        print("______________________")
        try:
            c_group = classification_groups[int(input("Select a CLASSIFICATION GROUP from the list above: "))]
        except (KeyError, IndexError, ValueError):
            print("Could not understand input. Insure input is of type int and in the range")
            c_group = None
        print("\n")

    return c_group


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

