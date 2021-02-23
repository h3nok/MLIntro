import re
from subprocess import Popen, PIPE
import configparser
import smtplib
import ssl
import email
import time
import os


class MachineMonitor:
    __computer = None
    _email_client = None

    def __init__(self, host_name=None, ini_file=None, email_ini=None):
        """
        must provide either host_name or ini_file
        @param host_name: host name of computer to monitor. this takes priority over ini
        @param ini_file: Ini file containing host_name of computer to monitor
        @param email_ini: If you want email notifications provide an email ini. see below
        """
        if host_name:
            self.__computer = host_name
        elif ini_file:
            config = configparser.ConfigParser()
            config.read(ini_file)
            assert config.has_section('machine_monitor') and config.has_option('machine_monitor', 'host_name')
            self.__computer = config.get('machine_monitor', 'host_name')
        else:
            raise BaseException("Must provide either host_name or ini_file")

        if email_ini:
            self._email_client = EmailClient(email_ini)

    def ping(self):
        """ Pings host_name and returns true if ping is received and false if not"""
        p = Popen('ping -n 1 ' + self.__computer, stdout=PIPE).stdout.read().decode("utf-8")
        # find number of sent and received messages or determine if unreachable.
        m = re.findall(r'Received = \d*|Sent = \d*', p)
        unreachable = re.findall(r'unreachable', p)
        # parse for just the number and make sure they are the same
        nums = [num for num in re.findall(r'\d+', " ".join(m))]

        # if any sent message was not received
        if len(nums) == 2 and nums[0] == nums[1] and len(unreachable) == 0:
            return True
        else:
            return False

    def is_up(self):
        """ If ping fails we will try a second time to verify that the system in down. if
        two True pings are received after the first failure we still return true"""
        if self.ping():
            return True
        else:
            return self.ping() and self.ping()

    def loop_check(self, interval=3, email_to=None):
        """
        Infinite loop to check status of computer host_name. Will email when ping fails
        if email_ini was provided. Otherwise it will just print. Will email once per shutdown
        @param interval: time interval between each ping
        @param email_to: recipient email address. If None it use default in ini.
        """
        last = True
        while True:
            time.sleep(interval)
            if self.is_up():
                print(f"{self.__computer}: Is Alive")
                last = True
                continue

            if last:
                if self._email_client:
                    self._email_client.send_email(email_to, f"{self.__computer}: Non-responsive.")
            print(f"{self.__computer}: Non-responsive.")
            last = False


class EmailClient:
    _section = 'email_client'

    _server = None
    _username = None
    _password = None
    _port = None

    _default_receiver = None

    def __init__(self, ini_file):
        """
        initialize email client. This work with boulder imaging email fine.
        for the ini file use: server=smtp.office365.com and port 587
        @param ini_file: email ini_file with server and login
        """
        # Read ini file and parse login info
        config = configparser.ConfigParser()
        config.read(ini_file)

        # The ini file must have the following 4 options
        assert config.has_section(self._section)
        assert config.has_option(self._section, 'server') and config.has_option(self._section, 'port') and \
            config.has_option(self._section, 'username') and config.has_option(self._section, 'password')

        self._server = config.get(self._section, 'server')
        self._username = config.get(self._section, 'username')
        self._password = config.get(self._section, 'password')
        self._port = config.getint(self._section, 'port')

        # load default if it exists
        if config.has_option(self._section, "default_receiver"):
            self._default_receiver = config.get(self._section, "default_receiver")

    def send_email(self, to=None, subject="", contents=""):
        """
        Send email to either to or default
        @param to: receiver. If None then default_receiver
        @param subject: subject of email
        @param contents: body of email
        """
        # construct the email
        msg = email.message.Message()
        msg.set_unixfrom('pymotw')
        msg["Subject"] = subject
        msg["From"] = self._username
        if to is None and self._default_receiver is not None:
            to = self._default_receiver
        elif to is None:
            raise BaseException("must supply param 'to' or default_receiver in ini")
        msg["To"] = to
        msg.set_payload(f'{contents}\n')

        # start email client to send
        context = ssl.create_default_context()
        with smtplib.SMTP(self._server, self._port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(self._username, self._password)
            server.sendmail(self._username, to, str(msg).encode('utf-8').replace(b'\n', b'\r\n'))


if __name__ == '__main__':
    config_file = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + r'\.config\machine_monitor.ini'
    wm = MachineMonitor(ini_file=config_file, email_ini=config_file)
    wm.loop_check()
