from tools.machine_monitor import MachineMonitor, EmailClient
import os


class TestMachineMonitor:
    def test_unknown_host(self):
        # This should result in unreachable
        mm = MachineMonitor(host_name="Unknown123")
        assert not mm.ping()
        assert not mm.is_up()

    def test_valid_host(self):
        mm = MachineMonitor(host_name="TigerShark_Soup")
        assert mm.ping()
        assert mm.is_up()


class TestEmailClient:
    def test_send_email(self):
        with open("temp.ini", "w+") as f:
            f.write("""[email_client]
                        server=smtp.office365.com
                        username=ijorquera@boulderimaging.com
                        password=<password here>
                        port=587
                        default_receiver=ijorquera@boulderimaging.com""")

        ec = EmailClient("temp.ini")
        ec.send_email(None, "This is a test email", "test body.")
        os.remove("temp.ini")
