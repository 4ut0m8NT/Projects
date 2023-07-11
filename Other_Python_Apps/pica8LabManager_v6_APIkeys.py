### Pica8 Lab Manager Example
### Neal Trieber, Pica8, Inc.
#####################################################
# * PRELIMINARY STEPS / PREP * ######################
#####################################################
# STEP 1) pip install -r requirements.txt --> included in repo!
# STEP 2) 
##########***********************************************
# NOTES: *      - USES STREAMLIT as a UI for Pica8 Credential Entry ONLY (Instead of the CLI)
#               - USES ANSIBLE (via BASH SHELL calls) to interact with Pica8/PicOS CLI
#               - USES Pica8's AmpCon REST APIs to interact with AmpCon: https://docs.pica8.com/display/ampcon/AmpCon+API+document
#               - SAMPLE Bash script provided to start your SLACKBOT and "bash source" your newly created SLACK Keys into SHELL variables
#               - Requirements.txt to help you auto-install all of the needed Python modules: pip install -r requirements.txt
##########***********************************************

from __future__ import (absolute_import, division, print_function)
import streamlit as st
from queue import Full
__metaclass__ = type
import re
import requests
import paramiko
import sys
import os, shutil, getopt
import logging
import glob
import io
from pathlib import Path
import types
import urllib
import SessionState
import subprocess
import json
import simplejson
from urllib.parse import urlparse
from subprocess import Popen, PIPE, STDOUT
import datetime
from datetime import datetime as date
import hashlib
import collections
import functools
import inspect
import textwrap
import time
import random
import string
from slack_sdk import WebClient
from slack_bolt import App, Say
from slack_bolt.adapter.socket_mode import SocketModeHandler
import getpass
from streamlit.legacy_caching.hashing import _CodeHasher
import collections
import functools
import inspect
import textwrap
from streamlit.server.server import Server
# from streamlit.scriptrunner.script_run_context
import time
import random
import string
try:
    from streamlit.scriptrunner.script_run_context import get_script_run_ctx as get_report_ctx 
    from streamlit.scriptrunner.script_run_context import add_script_run_ctx
except ModuleNotFoundError:
    # streamlit < 1.8
    try:
        from streamlit.scriptrunner.script_run_context import get_script_run_ctx  # type: ignore
    except ModuleNotFoundError:
        # streamlit < 1.4
        from streamlit.report_thread import (  # type: ignore
            get_script_run_ctx as get_report_ctx
        )

# try:
#     # Before Streamlit 0.65
#     from streamlit.ReportThread import get_report_ctx
#     from streamlit.server.Server import Server
# except ModuleNotFoundError:
#print ("FOOBAR")
# After Streamlit 0.65
# from streamlit.scriptrunner.script_run_context import get_report_ctx
from streamlit.server.server import Server
image1 = './pica8_logo_transparent.png'
image2 = './Lab_Manager.png'
st.set_page_config(page_title="Pica8 Lab Manager - SE LAB", page_icon=image1, layout="wide")
#st.set_page_config(page_title=None, page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.cache(persist=True, allow_output_mutation=True)
from streamlit import caching
import pysftp as sftp
import pysftp
import sys
logging.basicConfig(filename='pica8labmanager.log', filemode='w', level=logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger = logging.getLogger('mylogger')
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')
def my_handler(type, value, tb):
    logger.exception("Uncaught exception: {0}".format(str(value)))

# Install exception handler
sys.excepthook = my_handler

inputfile= ''
pica8apikey = ''
pica8instance = ''
bearerstr = ''
runlabresetbutton = ''
backupthelabbutton = ''
goldenlabbackupbutton = ''
# labreset = ''
datetimestamp = str(datetime.datetime.now()).replace(' ', '_')

if sys.version[0] == '2':
    sys.reload(sys)
    sys.setdefaultencoding("utf-8")

picosrequests = []

def setargs(bearerstr, pica8apikey, pica8instance, argv):
    try:
        opts, remainder = getopt.gnu_getopt(sys.argv[1:], 'i:k:h:', ['help=','pica8apikey=','pica8instance='])
    except getopt.GetoptError as err:
        print ('USAGE: streamlit run pica8SLACKBot.py -- - option: \n -i <FQDN or IP address of AmpCon - I.e. www.ampcon.com or 10.10.10.1)\n -k <Token/APIKey provided by AmpCon for authentication>\n',  err)
        sys.exit(2)
    if len(sys.argv) <= 1:
        # printbanner()
        print ('USAGE: streamlit run pica8SLACKBot.py -- - option: \n -i <FQDN or IP address of AmpCon - I.e. www.ampcon.com or 10.10.10.1)\n -k <Token/APIKey provided by AmpCon for authentication>\n')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h'):
            # printbanner()
            print ('USAGE: streamlit run pica8SLACKBot.py -- - option: \n -i <FQDN or IP address of AmpCon - I.e. www.ampcon.com or 10.10.10.1)\n -k <Token/APIKey provided by AmpCon for authentication>\n')
            sys.exit()
        if opt in ('-k'):
            pica8apikey = (arg)
            print ("The ApiKey/Token is: " + str(pica8apikey))
            bearerstr = 'Bearer ' + str(pica8apikey)
            #return inputfile
        if opt in ('-i'):
            pica8instance = (arg)
            print ("The AmpCon Instance being used is: " + str(pica8instance))
            #return inputfile
    return bearerstr, pica8apikey, pica8instance, argv
 

 
bearerstr, pica8apikey, pica8instance, sys.argv = setargs(bearerstr, pica8apikey, pica8instance, sys.argv)


# Simple shell-based Alternative to perform Login to AmpCon for automatic API key-generation, alternatively a FLASH or streamlit interface could be used to gather the info as well
# this can be taken as "switches/command-line arguments" as well..
# print ("Please Enter your AmpCon Username:")
# username = input()
# password = getpass.getpass(prompt='Please Enter your AmpCon Password: ', stream=None)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

def settokenheaders (pica8apikey):
    
    tokenheaders = {
        'Accept': 'application/json',
        'Authorization': 'Bearer {}'.format(pica8apikey), 
        'Content-Type': 'application/json',
    }
    
    return tokenheaders


local_css("labman.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

#icon("search")
#button_clicked = st.button("OK")




pica8apiurl =  "https://" + pica8instance + "/api"

col1, col2 = st.columns(2)
caption = "AmpCon Login: Lab Manager Needs to login into AmpCon. Please Enter Your AmpCon Credentials."

col1.image(image1, caption=None, width=None, use_column_width=True,
               clamp=False, channels='RGB', output_format='auto')
col1.image(image2, caption=None, width=None, use_column_width=True,
               clamp=False, channels='RGB', output_format='auto')

col1.title("Pica8 AmpCon Lab Manager")
col1.subheader("Keeping the Lab 'Demo-Ready'.....")

     
st.sidebar.image(image1, caption=caption, width=250, use_column_width=False,
                 clamp=False, channels='RGB', output_format='auto')
st.sidebar.title("Pica8 AmpCon Authentication\n & Secure API Key Generator for Lab Management")
username = st.sidebar.text_input('username')
password = st.sidebar.text_input('password', type='password')
loggedin = ''
response = ''

tokenurl = "https://" + pica8instance + "/token"

#status = subprocess.check_output("systemctl show -p ActiveState --value abc", shell=True)
restartlabmanagerbutton = col2.button('RESTART LAB MANAGER')
col2.write ('PRESS RESTART BUTTON IF \"switchconfigs\" error has occurred (Lab Manager may periodically lose connection to AmpCon and need token refresh)')
if restartlabmanagerbutton:
    labmanager_restart_command = "systemctl restart pica8labmanager"
    subprocess.call(['bash', '-c', labmanager_restart_command])

class _SessionState:
    
    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()

    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False

        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(
            self._state["data"], None)


def get_session_id():
    ctx = get_report_ctx()
    return ctx.session_id



def cache_on_sidebarbutton_press(label, **cache_kwargs):
    """Function decorator to memorize function executions.
    Parameters
    ----------
    label : str
        The label for the button to display prior to running the cached funnction.
    cache_kwargs : Dict[Any, Any]
        Additional parameters (such as show_spinner) to pass into the underlying @st.cache decorator.
    Example
    -------
    This show how you could write a username/password tester:
    >>> @cache_on_button_press('Authenticate')
    ... def authenticate(username, password):
    ...     return username == "buddha" and password == "s4msara"
    ...
    ... username = st.text_input('username')
    ... password = st.text_input('password')
    ...
    ... if authenticate(username, password):
    ...     st.success('Logged in.')
    ... else:
    ...     st.error('Incorrect username or password')
    """
    internal_cache_kwargs = dict(cache_kwargs)
    internal_cache_kwargs['allow_output_mutation'] = True
    internal_cache_kwargs['show_spinner'] = False

    def function_decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            @st.cache(**internal_cache_kwargs)
            def get_cache_entry(func, args, kwargs):
                class ButtonCacheEntry:
                    def __init__(self):
                        self.evaluated = False
                        self.return_value = None

                    def evaluate(self):
                        self.evaluated = True
                        self.return_value = func(*args, **kwargs)
                return ButtonCacheEntry()
            cache_entry = get_cache_entry(func, args, kwargs)
            if not cache_entry.evaluated:
                if st.sidebar.button(label):
                    cache_entry.evaluate()
                else:
                    raise st.stop()
            return cache_entry.return_value
        return wrapped_func
    return function_decorator


def cache_on_button_press(label, **cache_kwargs):
    """Function decorator to memoize function executions.
    Parameters
    ----------
    label : str
        The label for the button to display prior to running the cached funnction.
    cache_kwargs : Dict[Any, Any]
        Additional parameters (such as show_spinner) to pass into the underlying @st.cache decorator.
    Example
    -------
    This show how you could write a username/password tester:
    >>> @cache_on_button_press('Authenticate')
    ... def authenticate(username, password):
    ...     return username == "buddha" and password == "s4msara"
    ...
    ... username = st.text_input('username')
    ... password = st.text_input('password')
    ...
    ... if authenticate(username, password):
    ...     st.success('Logged in.')
    ... else:
    ...     st.error('Incorrect username or password')
    """
    internal_cache_kwargs = dict(cache_kwargs)
    internal_cache_kwargs['allow_output_mutation'] = True
    internal_cache_kwargs['show_spinner'] = False

    def function_decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            @st.cache(**internal_cache_kwargs)
            def get_cache_entry(func, args, kwargs):
                class ButtonCacheEntry:
                    def __init__(self):
                        self.evaluated = False
                        self.return_value = None

                    def evaluate(self):
                        self.evaluated = True
                        self.return_value = func(*args, **kwargs)
                return ButtonCacheEntry()
            cache_entry = get_cache_entry(func, args, kwargs)
            if not cache_entry.evaluated:
                if st.button(label):
                    cache_entry.evaluate()
                else:
                    raise st.stop()
            return cache_entry.return_value
        return wrapped_func
    return function_decorator
def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")

    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state

authpayload = json.dumps({
        "username": username,
        "password": password
    })


@cache_on_sidebarbutton_press('LOGIN')
def LOGIN(username, password, loggedin, pica8apikey, authpayload, runlabresetbutton, backupthelabbutton, goldenlabbackupbutton):
    payload = json.dumps({
        "username": username,
        "password": password
    })
    tokenheaders = settokenheaders(pica8apikey)
    keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    pica8apikey = keyresponse.text
    tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("POST", tokenurl, headers=tokenheaders, data=payload, verify=False)
    try:
        responsetext = (json.loads(response.text))
        for msg in responsetext:
            result = (responsetext[msg])
            if (result == "Username or Password is incorrect"):
                loggedin = "failed"
                st.sidebar.error("Login Failed: Username or Password is incorrect")
                col1.error("Login Failed: Username or Password is incorrect")
    except:
        loggedin = "success"
        pica8apikey = response.text
        print ("the key to our success: " + pica8apikey)
        st.sidebar.success("Login Successful!")
        st.sidebar.success("Logged In as {}".format(username))
        col1apikeyexpander = col1.expander("Retrieved your API Key from AmpCon:")
        col1apikeyexpander.write(pica8apikey)    
        #col1.write("Retrieved your API Key from AmpCon: " + pica8apikey)
        col1.write("********************************************")
        col1.subheader("API KEY GENERATED SUCCESSFULLY!")
        col1.write("********************************************")
        col1.subheader("SE Lab Manager IS READY!")
        col1.write("********************************************")
        for key in st.session_state.keys():
            del st.session_state[key]
       
        
                 
    print ("AmpCon login Status: " + loggedin)

    return username, password, loggedin, pica8apikey, authpayload, runlabresetbutton, backupthelabbutton, goldenlabbackupbutton

def backupthelablocally(col2expander, username, password, loggedin, pica8apikey, authpayload, tokenheaders, pica8apiurl, sgselector, allchecked, deviceschosen, switchgroup_devices, custombackup_tag, datetimestamp, sftp, deviceslist):
    col2expander.write ("Backing up The Lab Now.....Please Standby.....")
    switchListing = {}
    configListing = {}
    if (deviceschosen):
        sgselector = []
        for device in deviceschosen:
            jobspayload=""
            devicebackupurl = pica8apiurl + "/backup_config/" + device
            print (devicebackupurl)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse)
    if (sgselector):
        print ("SG SELECTED!")
        print("**************************")
        print (sgselector)
        print("**************************")
        deviceschosen = []
        sgdevices = []
        for switchgroupname in sgselector:
            print ("Backing up each device in: " + switchgroupname)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("GET", groupswitchlisturl, headers=tokenheaders, data=authpayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse)
            switchgroupslist = json.loads(switchconfigsresponse)
            for switchgroup in switchgroupslist:
                #print (switchgroup)
                #print (switchgroup['name'])
                #switch_groups.append(str(switchgroup['name']))
                if switchgroupname ==  str((switchgroup['name'])):
                    sgdevices = ((switchgroup['sn']))
                    print (sgdevices)
            for device in sgdevices:
                jobspayload=""
                print ("Backing up Device: " + device)
                devicebackupurl = pica8apiurl + "/backup_config/" + device
                print (devicebackupurl)
                keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
                pica8apikey = keyresponse.text
                tokenheaders = settokenheaders(pica8apikey)
                response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
                switchconfigsresponse = response.text
            print (switchconfigsresponse)
    if (allchecked == True):
        deviceschosen = []
        sgselector = []
        alldevices = []
        for device in deviceslist:
            chosendevice = device
            chosendevicestr = str(chosendevice)
            print('Looking up SN For: ' + chosendevicestr)
            print('********************************************\n')
            devserialnum = str(deviceslist.get(chosendevicestr))
            alldevices.append(devserialnum)
        for device in alldevices:
            jobspayload=""
            devicebackupurl = pica8apiurl + "/backup_config/" + device
            print (devicebackupurl)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse)    
    print ("Backup headers: " + str(tokenheaders))
    print ("The key: " + str(pica8apikey))
    print ("All Checked Status: " + str(allchecked))
    #print ("Devices Chosen: " + deviceschosen)
    playbookname = "FULL_BACKUP_SET-SAVE"
    #playbookname = "Backup_Config_local"
    playbookfile = "playbook.yml"
    playbookfolder = playbookname + "/" + playbookfile
    payload = json.dumps({
        "playbook_name": playbookname,
        "playbook_dir": playbookfolder,
        "switches": deviceschosen,
        "switch_checkall": allchecked,
        "group_list": sgselector,
        "vars": {"custombackup_tag" : custombackup_tag},
        "scheduled": {
            "type": "DIRECT",
            "params": {}
        }
    })
    print (payload)
    # payload = json.dumps({
    #     "config_name": configtodeploy,
    #     "switch_checked": {
    #         "checkall": False,
    #         "checkedNodes": [
    #         {
    #             "sn": switchtodeployserial
    #         }
    #         ],
    #         "uncheckedNodes": []
    #     },
    #     "group_checked": {
    #         "checkall": False,d
    #         "checkedNodes": [],
    #         "uncheckedNodes": []
    #     }
    #     })
    # keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    # pica8apikey = keyresponse.text
    # tokenheaders = settokenheaders(pica8apikey)
    # response = requests.request("POST", deploymenturl, headers=headers, data=payload, verify=False)
    # print(response.text)
    ampconplaybookurl = pica8apiurl + "/ansible/playbooks/run"
    print (ampconplaybookurl)
    keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    pica8apikey = keyresponse.text
    tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("POST", ampconplaybookurl, headers=tokenheaders, data=payload, verify=False)
    switchconfigsresponse = response.text
    print (switchconfigsresponse)
    playbookstatus = json.loads(switchconfigsresponse)
    col2expander.write('Backing up Devices with Configs with Today\'s date to /home/admin on the DEVICE!...Standby')
    # for SwitchConfig in switchconfigs:
    print ('*********************************\n')
    col2expander.write ('*********************************\n')
    # playbookinfo = str(playbookstatus['info'])
    # print("Info:" + playbookinfo)
    # col2expander.write("Info:" + playbookinfo)
    playbookjob = str(playbookstatus['job_name'])
    print("Job:" + playbookjob)
    col2expander.write("Job:" + playbookjob)
    playbookstats = str(playbookstatus['status'])
    print("Status:" + playbookstats)
    col2expander.write("Status:" + playbookstats)
    print ('*********************************\n')
    col2expander.write ('*********************************\n')
    ampconplaybookjobsurl = pica8apiurl + "/ansible/jobs"
    print (ampconplaybookjobsurl)
    jobspayload=""
    jobstatus="IDLE"
    while (jobstatus == "IDLE") or (jobstatus == "RUNNING"):
        keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
        pica8apikey = keyresponse.text
        tokenheaders = settokenheaders(pica8apikey)
        response = requests.request("GET", ampconplaybookjobsurl, headers=tokenheaders, data=jobspayload, verify=False)
        switchconfigsresponse = response.text
        #print (switchconfigsresponse)
        playbookjobsstatus = json.loads(switchconfigsresponse)
        for playbookjobstatus in playbookjobsstatus:
            # print ("ELEMENT: ")
            # print ("****************************")
            # print (playbookjobstatus)
            # print ("****************************")
            # print ("STATUS:" + str(jobstatus))
            # print ("****************************")
            if ( (str(playbookjobstatus['name'])) == playbookjob):
                jobstatus = str(playbookjobstatus['status'])
                print("Job:" + playbookjob)
                print ("Found:" + str(playbookjobstatus['name']))
                print ("STATUS:" + str(jobstatus))
                col2expander.write("Waiting for Backups to Complete......Standby.....")
    keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    pica8apikey = keyresponse.text
    tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("GET", ampconplaybookjobsurl, headers=tokenheaders, data=jobspayload, verify=False)
    switchconfigsresponse = response.text
    #print (switchconfigsresponse)
    playbookjobsstatus = json.loads(switchconfigsresponse)
    for playbookjobstatus in playbookjobsstatus:
        # print ("ELEMENT: ")
        # print ("****************************")
        # print (playbookjobstatus)
        # print ("****************************")
        # print ("STATUS:" + str(jobstatus))
        # print ("****************************")
        if ( (str(playbookjobstatus['name'])) == playbookjob):    
            results = str(playbookjobstatus['results'])
            jobstatus = str(playbookjobstatus['status'])
            print ("****************************")
            print (playbookjobstatus)
            print ("****************************")    
            results = str(playbookjobstatus['results'])
            zipfile_regex = re.compile('(/home/admin/Config_Backups/(.*)/(.*\.zip))')
            fileparts = re.findall(zipfile_regex, results)
            zipfilepath = str(fileparts[0])
            #resultsdata = json.loads(results)
            # for result in results:
            #     print (result)
            jobstatus = str(playbookjobstatus['status'])
            print ("STATUS:" + str(jobstatus))
            #print ("RESULTS:" + results)
            # zipfile_regex = re.compile('(/home/admin/Config_Backups/(.*)/(.*\.zip))')
            zipfile_regex = re.compile('(/home/admin/Config_Backups/(.*)/(.*\.zip))')
            fileparts = re.findall(zipfile_regex, results)
            print ("*************PARTS IS PARTS**************************")
            print (fileparts)
            print ("*************PARTS IS PARTS**************************")
            filename = str(fileparts[0][2])
            downloadpath = "/home/admin/Config_Backups/" + custombackup_tag + "/" + filename
            print(downloadpath)
            myfolder = "./Config_Backups/"
            #savepath =  "/Users/ntrieber/Documents/Scripts/Notebooks/Config_Backups/" + filename
            savepath = myfolder + filename
            print(savepath)
            sftp.get(downloadpath, savepath)
            t.close()
            with open(savepath, 'rb') as f:
                col1.download_button('Download Zip', f, file_name=filename)
            col2expander.write(results)
            col2expander.write ('*********************************\n')
            col1.subheader("LAB BACKUP COMPLETE!")
            

def labreset(col2expander, username, password, loggedin, pica8apikey, authpayload, tokenheaders, pica8apiurl, sgselector, allchecked, deviceschosen, switchgroup_devices, custombackup_tag, datetimestamp, sftp, deviceslist):
    col2expander.write ("Resetting Switches.....Please Standby.....")
    switchListing = {}
    configListing = {}
    if (deviceschosen):
        sgselector = []
        for device in deviceschosen:
            jobspayload=""
            devicebackupurl = pica8apiurl + "/backup_config/" + device
            print (devicebackupurl)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse)
    if (sgselector):
        print ("SG SELECTED!")
        print("**************************")
        print (sgselector)
        print("**************************")
        deviceschosen = []
        sgdevices = []
        for switchgroupname in sgselector:
            print ("Backing up each device in: " + switchgroupname)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("GET", groupswitchlisturl, headers=tokenheaders, data=authpayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse)
            switchgroupslist = json.loads(switchconfigsresponse)
            for switchgroup in switchgroupslist:
                #print (switchgroup)
                #print (switchgroup['name'])
                #switch_groups.append(str(switchgroup['name']))
                if switchgroupname ==  str((switchgroup['name'])):
                    sgdevices = ((switchgroup['sn']))
                    print (sgdevices)
            for device in sgdevices:
                jobspayload=""
                print ("Backing up Device: " + device)
                devicebackupurl = pica8apiurl + "/backup_config/" + device
                print (devicebackupurl)
                keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
                pica8apikey = keyresponse.text
                tokenheaders = settokenheaders(pica8apikey)
                response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
                switchconfigsresponse = response.text
                print (switchconfigsresponse)
    if (allchecked == True):
        deviceschosen = []
        sgselector = []
        alldevices = []
        for device in deviceslist:
            chosendevice = device
            chosendevicestr = str(chosendevice)
            print('Looking up SN For: ' + chosendevicestr)
            print('********************************************\n')
            devserialnum = str(deviceslist.get(chosendevicestr))
            alldevices.append(devserialnum)
        for device in alldevices:
            jobspayload=""
            devicebackupurl = pica8apiurl + "/backup_config/" + device
            print (devicebackupurl)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse) 
    print ("Lab reset headers: " + str(tokenheaders))
    print ("The key: " + str(pica8apikey))
    playbookname = "RESET_Config_for_DEMO"
    playbookfile = "playbook.yml"
    playbookfolder = playbookname + "/" + playbookfile
    payload = json.dumps({
        "playbook_name": playbookname,
        "playbook_dir": playbookfolder,
        "switches": deviceschosen,
        "switch_checkall": allchecked,
        "group_list": sgselector,
        "vars": {"custombackup_tag" : custombackup_tag},
        "scheduled": {
            "type": "DIRECT",
            "params": {}
        }
    })
    print (payload)
    # payload = json.dumps({
    #     "config_name": configtodeploy,
    #     "switch_checked": {
    #         "checkall": False,
    #         "checkedNodes": [
    #         {
    #             "sn": switchtodeployserial
    #         }
    #         ],
    #         "uncheckedNodes": []
    #     },
    #     "group_checked": {
    #         "checkall": False,d
    #         "checkedNodes": [],
    #         "uncheckedNodes": []
    #     }
    #     })
    # keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    # pica8apikey = keyresponse.text
    # tokenheaders = settokenheaders(pica8apikey)
    # response = requests.request("POST", deploymenturl, headers=headers, data=payload, verify=False)
    # print(response.text)
    ampconplaybookurl = pica8apiurl + "/ansible/playbooks/run"
    print (ampconplaybookurl)
    keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    pica8apikey = keyresponse.text
    tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("POST", ampconplaybookurl, headers=tokenheaders, data=payload, verify=False)
    switchconfigsresponse = response.text
    print (switchconfigsresponse)
    playbookstatus = json.loads(switchconfigsresponse)
    col2expander.write('Re-Setting Devices to Golden Demo Configs...Standby')
    # for SwitchConfig in switchconfigs:
    print ('*********************************\n')
    col2expander.write ('*********************************\n')
    # playbookinfo = str(playbookstatus['info'])
    # print("Info:" + playbookinfo)
    # col2expander.write("Info:" + playbookinfo)
    playbookjob = str(playbookstatus['job_name'])
    print("Job:" + playbookjob)
    col2expander.write("Job:" + playbookjob)
    playbookstats = str(playbookstatus['status'])
    print("Status:" + playbookstats)
    col2expander.write("Status:" + playbookstats)
    print ('*********************************\n')
    col2expander.write ('*********************************\n')
    ampconplaybookjobsurl = pica8apiurl + "/ansible/jobs"
    print (ampconplaybookjobsurl)
    jobspayload=""
    jobstatus="IDLE"
    while (jobstatus == "IDLE") or (jobstatus == "RUNNING"):
        keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
        pica8apikey = keyresponse.text
        tokenheaders = settokenheaders(pica8apikey)
        response = requests.request("GET", ampconplaybookjobsurl, headers=tokenheaders, data=jobspayload, verify=False)
        switchconfigsresponse = response.text
        #print (switchconfigsresponse)
        playbookjobsstatus = json.loads(switchconfigsresponse)
        for playbookjobstatus in playbookjobsstatus:
            # print ("ELEMENT: ")
            # print ("****************************")
            # print (playbookjobstatus)
            # print ("****************************")
            # print ("STATUS:" + str(jobstatus))
            # print ("****************************")
            if ( (str(playbookjobstatus['name'])) == playbookjob):
                jobstatus = str(playbookjobstatus['status'])
                print("Job:" + playbookjob)
                print ("Found:" + str(playbookjobstatus['name']))
                print ("STATUS:" + str(jobstatus))
                col2expander.write("Waiting for Lab Reset to Complete......Standby.....")
    keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    pica8apikey = keyresponse.text
    tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("GET", ampconplaybookjobsurl, headers=tokenheaders, data=jobspayload, verify=False)
    switchconfigsresponse = response.text
    #print (switchconfigsresponse)
    playbookjobsstatus = json.loads(switchconfigsresponse)
    for playbookjobstatus in playbookjobsstatus:
        # print ("ELEMENT: ")

        # print ("STATUS:" + str(jobstatus))
        # print ("****************************")
        if ( (str(playbookjobstatus['name'])) == playbookjob):
            print ("****************************")
            print (playbookjobstatus)
            print ("****************************")    
            results = str(playbookjobstatus['results'])
            jobstatus = str(playbookjobstatus['status'])
            print ("STATUS:" + str(jobstatus))
            #print ("RESULTS:" + results)
            zipfile_regex = re.compile('(/home/admin/Config_Backups/(.*)/(.*\.zip))')
            fileparts = re.findall(zipfile_regex, results)
            print ("*************PARTS IS PARTS**************************")
            print (fileparts)
            print ("*************PARTS IS PARTS**************************")
            filename = str(fileparts[0][2])
            downloadpath = "/home/admin/Config_Backups/" + custombackup_tag + "/" + filename
            print(downloadpath)
            myfolder = "./Config_Backups/"
            #savepath =  "/Users/ntrieber/Documents/Scripts/Notebooks/Config_Backups/" + filename
            savepath = myfolder + filename
            print(savepath)
            sftp.get(downloadpath, savepath)
            t.close()
            col2expander.write(results)
            col2expander.write ('*********************************\n')
            col1.subheader("LAB RESET COMPLETE AND READY FOR DEMO!")
            col1.subheader("SOME DEVICES MAY REQUIRE REBOOT!")
            with open(savepath, 'rb') as f:
                col1.download_button('Download Zip', f, file_name=filename)


def labbasicconfigreset(col2expander, username, password, loggedin, pica8apikey, authpayload, tokenheaders, pica8apiurl, sgselector, allchecked, deviceschosen, switchgroup_devices, custombackup_tag, datetimestamp, sftp, deviceslist):
    col2expander.write ("Resetting Switches.....Please Standby.....")
    switchListing = {}
    configListing = {}
    if (deviceschosen):
        sgselector = []
        for device in deviceschosen:
            jobspayload=""
            devicebackupurl = pica8apiurl + "/backup_config/" + device
            print (devicebackupurl)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse)
    if (sgselector):
        print ("SG SELECTED!")
        print("**************************")
        print (sgselector)
        print("**************************")
        deviceschosen = []
        sgdevices = []
        for switchgroupname in sgselector:
            print ("Backing up each device in: " + switchgroupname)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("GET", groupswitchlisturl, headers=tokenheaders, data=authpayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse)
            switchgroupslist = json.loads(switchconfigsresponse)
            for switchgroup in switchgroupslist:
                #print (switchgroup)
                #print (switchgroup['name'])
                #switch_groups.append(str(switchgroup['name']))
                if switchgroupname ==  str((switchgroup['name'])):
                    sgdevices = ((switchgroup['sn']))
                    print (sgdevices)
            for device in sgdevices:
                jobspayload=""
                print ("Backing up Device: " + device)
                devicebackupurl = pica8apiurl + "/backup_config/" + device
                print (devicebackupurl)
                keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
                pica8apikey = keyresponse.text
                tokenheaders = settokenheaders(pica8apikey)
                response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
                switchconfigsresponse = response.text
                print (switchconfigsresponse)
    if (allchecked == True):
        deviceschosen = []
        sgselector = []
        alldevices = []
        for device in deviceslist:
            chosendevice = device
            chosendevicestr = str(chosendevice)
            print('Looking up SN For: ' + chosendevicestr)
            print('********************************************\n')
            devserialnum = str(deviceslist.get(chosendevicestr))
            alldevices.append(devserialnum)
        for device in alldevices:
            jobspayload=""
            devicebackupurl = pica8apiurl + "/backup_config/" + device
            print (devicebackupurl)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse) 
    print ("Lab reset headers: " + str(tokenheaders))
    print ("The key: " + str(pica8apikey))
    playbookname = "Basic_Config_Load"
    playbookfile = "playbook.yml"
    playbookfolder = playbookname + "/" + playbookfile
    payload = json.dumps({
        "playbook_name": playbookname,
        "playbook_dir": playbookfolder,
        "switches": deviceschosen,
        "switch_checkall": allchecked,
        "group_list": sgselector,
        "vars": {"custombackup_tag" : custombackup_tag},
        "scheduled": {
            "type": "DIRECT",
            "params": {}
        }
    })
    print (payload)
    # payload = json.dumps({
    #     "config_name": configtodeploy,
    #     "switch_checked": {
    #         "checkall": False,
    #         "checkedNodes": [
    #         {
    #             "sn": switchtodeployserial
    #         }
    #         ],
    #         "uncheckedNodes": []
    #     },
    #     "group_checked": {
    #         "checkall": False,d
    #         "checkedNodes": [],
    #         "uncheckedNodes": []
    #     }
    #     })
    # keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    # pica8apikey = keyresponse.text
    # tokenheaders = settokenheaders(pica8apikey)
    # response = requests.request("POST", deploymenturl, headers=headers, data=payload, verify=False)
    # print(response.text)
    ampconplaybookurl = pica8apiurl + "/ansible/playbooks/run"
    print (ampconplaybookurl)
    keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    pica8apikey = keyresponse.text
    tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("POST", ampconplaybookurl, headers=tokenheaders, data=payload, verify=False)
    switchconfigsresponse = response.text
    print (switchconfigsresponse)
    playbookstatus = json.loads(switchconfigsresponse)
    col2expander.write('Re-Setting Devices to Golden Demo Configs...Standby')
    # for SwitchConfig in switchconfigs:
    print ('*********************************\n')
    col2expander.write ('*********************************\n')
    # playbookinfo = str(playbookstatus['info'])
    # print("Info:" + playbookinfo)
    # col2expander.write("Info:" + playbookinfo)
    playbookjob = str(playbookstatus['job_name'])
    print("Job:" + playbookjob)
    col2expander.write("Job:" + playbookjob)
    playbookstats = str(playbookstatus['status'])
    print("Status:" + playbookstats)
    col2expander.write("Status:" + playbookstats)
    print ('*********************************\n')
    col2expander.write ('*********************************\n')
    ampconplaybookjobsurl = pica8apiurl + "/ansible/jobs"
    print (ampconplaybookjobsurl)
    jobspayload=""
    jobstatus="IDLE"
    while (jobstatus == "IDLE") or (jobstatus == "RUNNING"):
        keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
        pica8apikey = keyresponse.text
        tokenheaders = settokenheaders(pica8apikey)
        response = requests.request("GET", ampconplaybookjobsurl, headers=tokenheaders, data=jobspayload, verify=False)
        switchconfigsresponse = response.text
        #print (switchconfigsresponse)
        playbookjobsstatus = json.loads(switchconfigsresponse)
        for playbookjobstatus in playbookjobsstatus:
            # print ("ELEMENT: ")
            # print ("****************************")
            # print (playbookjobstatus)
            # print ("****************************")
            # print ("STATUS:" + str(jobstatus))
            # print ("****************************")
            if ( (str(playbookjobstatus['name'])) == playbookjob):
                jobstatus = str(playbookjobstatus['status'])
                print("Job:" + playbookjob)
                print ("Found:" + str(playbookjobstatus['name']))
                print ("STATUS:" + str(jobstatus))
                col2expander.write("Waiting for Lab Reset to Complete......Standby.....")
    keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    pica8apikey = keyresponse.text
    tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("GET", ampconplaybookjobsurl, headers=tokenheaders, data=jobspayload, verify=False)
    switchconfigsresponse = response.text
    #print (switchconfigsresponse)
    playbookjobsstatus = json.loads(switchconfigsresponse)
    for playbookjobstatus in playbookjobsstatus:
        # print ("ELEMENT: ")

        # print ("STATUS:" + str(jobstatus))
        # print ("****************************")
        if ( (str(playbookjobstatus['name'])) == playbookjob):
            print ("****************************")
            print (playbookjobstatus)
            print ("****************************")    
            results = str(playbookjobstatus['results'])
            jobstatus = str(playbookjobstatus['status'])
            print ("STATUS:" + str(jobstatus))
            #print ("RESULTS:" + results)
            zipfile_regex = re.compile('(/home/admin/Config_Backups/(.*)/(.*\.zip))')
            fileparts = re.findall(zipfile_regex, results)
            print ("*************PARTS IS PARTS**************************")
            print (fileparts)
            print ("*************PARTS IS PARTS**************************")
            filename = str(fileparts[0][2])
            downloadpath = "/home/admin/Config_Backups/" + custombackup_tag + "/" + filename
            print(downloadpath)
            myfolder = "./Config_Backups/"
            #savepath =  "/Users/ntrieber/Documents/Scripts/Notebooks/Config_Backups/" + filename
            savepath = myfolder + filename
            print(savepath)
            sftp.get(downloadpath, savepath)
            t.close()
            col2expander.write(results)
            col2expander.write ('*********************************\n')
            col1.subheader("LAB RESET COMPLETE AND READY FOR DEMO!")
            col1.subheader("SOME DEVICES MAY REQUIRE REBOOT!")
            with open(savepath, 'rb') as f:
                col1.download_button('Download Zip', f, file_name=filename)



def labrestore(col2expander, username, password, loggedin, pica8apikey, authpayload, tokenheaders, pica8apiurl, sgselector, allchecked, deviceschosen, switchgroup_devices, custombackup_tag, customrestore_tag, dateselected, datetimestamp, sftp, deviceslist):
    col2expander.write ("Resetting Switches.....Please Standby.....")
    if not (custombackup_tag):
        custombackup_tag = "current_lab_config_" + datetimestamp
    switchListing = {}
    configListing = {}
    if (deviceschosen):
        sgselector = []
        for device in deviceschosen:
            jobspayload=""
            devicebackupurl = pica8apiurl + "/backup_config/" + device
            print (devicebackupurl)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse)
    if (sgselector):
        print ("SG SELECTED!")
        print("**************************")
        print (sgselector)
        print("**************************")
        deviceschosen = []
        sgdevices = []
        for switchgroupname in sgselector:
            print ("Backing up each device in: " + switchgroupname)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("GET", groupswitchlisturl, headers=tokenheaders, data=authpayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse)
            switchgroupslist = json.loads(switchconfigsresponse)
            for switchgroup in switchgroupslist:
                #print (switchgroup)
                #print (switchgroup['name'])
                #switch_groups.append(str(switchgroup['name']))
                if switchgroupname ==  str((switchgroup['name'])):
                    sgdevices = ((switchgroup['sn']))
                    print (sgdevices)
            for device in sgdevices:
                jobspayload=""
                print ("Backing up Device: " + device)
                devicebackupurl = pica8apiurl + "/backup_config/" + device
                print (devicebackupurl)
                keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
                pica8apikey = keyresponse.text
                tokenheaders = settokenheaders(pica8apikey)
                response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
                switchconfigsresponse = response.text
                print (switchconfigsresponse)
    if (allchecked == True):
        deviceschosen = []
        sgselector = []
        alldevices = []
        for device in deviceslist:
            chosendevice = device
            chosendevicestr = str(chosendevice)
            print('Looking up SN For: ' + chosendevicestr)
            print('********************************************\n')
            devserialnum = str(deviceslist.get(chosendevicestr))
            alldevices.append(devserialnum)
        for device in alldevices:
            jobspayload=""
            devicebackupurl = pica8apiurl + "/backup_config/" + device
            print (devicebackupurl)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse) 
    print ("Lab reset headers: " + str(tokenheaders))
    print ("The key: " + str(pica8apikey))
    playbookname = "RESET_Config_To_CustomTag"
    playbookfile = "playbook.yml"
    playbookfolder = playbookname + "/" + playbookfile
    backup_date = str(dateselected)
    print (backup_date)
    payload = json.dumps({
        "playbook_name": playbookname,
        "playbook_dir": playbookfolder,
        "switches": deviceschosen,
        "switch_checkall": allchecked,
        "group_list": sgselector,
        "vars": {"custombackup_tag" : custombackup_tag, 
                 "customrestore_tag" : customrestore_tag,
                 "backup_date" : backup_date},
        "scheduled": {
            "type": "DIRECT",
            "params": {}
        }
    })
    print (payload)
    # payload = json.dumps({
    #     "config_name": configtodeploy,
    #     "switch_checked": {
    #         "checkall": False,
    #         "checkedNodes": [
    #         {
    #             "sn": switchtodeployserial
    #         }
    #         ],
    #         "uncheckedNodes": []
    #     },
    #     "group_checked": {
    #         "checkall": False,d
    #         "checkedNodes": [],
    #         "uncheckedNodes": []
    #     }
    #     })
    # keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    # pica8apikey = keyresponse.text
    # tokenheaders = settokenheaders(pica8apikey)
    # response = requests.request("POST", deploymenturl, headers=headers, data=payload, verify=False)
    # print(response.text)
    ampconplaybookurl = pica8apiurl + "/ansible/playbooks/run"
    print (ampconplaybookurl)
    keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    pica8apikey = keyresponse.text
    tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("POST", ampconplaybookurl, headers=tokenheaders, data=payload, verify=False)
    switchconfigsresponse = response.text
    print (switchconfigsresponse)
    playbookstatus = json.loads(switchconfigsresponse)
    col2expander.write('Re-Setting Devices to ' + customrestore_tag + ' Configs...Standby')
    # for SwitchConfig in switchconfigs:
    print ('*********************************\n')
    col2expander.write ('*********************************\n')
    # playbookinfo = str(playbookstatus['info'])
    # print("Info:" + playbookinfo)
    # col2expander.write("Info:" + playbookinfo)
    playbookjob = str(playbookstatus['job_name'])
    print("Job:" + playbookjob)
    col2expander.write("Job:" + playbookjob)
    playbookstats = str(playbookstatus['status'])
    print("Status:" + playbookstats)
    col2expander.write("Status:" + playbookstats)
    print ('*********************************\n')
    col2expander.write ('*********************************\n')
    ampconplaybookjobsurl = pica8apiurl + "/ansible/jobs"
    print (ampconplaybookjobsurl)
    jobspayload=""
    jobstatus="IDLE"
    while (jobstatus == "IDLE") or (jobstatus == "RUNNING"):
        keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
        pica8apikey = keyresponse.text
        tokenheaders = settokenheaders(pica8apikey)
        response = requests.request("GET", ampconplaybookjobsurl, headers=tokenheaders, data=jobspayload, verify=False)
        switchconfigsresponse = response.text
        #print (switchconfigsresponse)
        playbookjobsstatus = json.loads(switchconfigsresponse)
        for playbookjobstatus in playbookjobsstatus:
            # print ("ELEMENT: ")
            # print ("****************************")
            # print (playbookjobstatus)
            # print ("****************************")
            # print ("STATUS:" + str(jobstatus))
            # print ("****************************")
            if ( (str(playbookjobstatus['name'])) == playbookjob):
                jobstatus = str(playbookjobstatus['status'])
                print("Job:" + playbookjob)
                print ("Found:" + str(playbookjobstatus['name']))
                print ("STATUS:" + str(jobstatus))
                col2expander.write("Waiting for Lab Reset to Complete......Standby.....")
    keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    pica8apikey = keyresponse.text
    tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("GET", ampconplaybookjobsurl, headers=tokenheaders, data=jobspayload, verify=False)
    switchconfigsresponse = response.text
    #print (switchconfigsresponse)
    playbookjobsstatus = json.loads(switchconfigsresponse)
    for playbookjobstatus in playbookjobsstatus:
        # print ("ELEMENT: ")
        # print ("****************************")
        # print (playbookjobstatus)
        # print ("****************************")
        # print ("STATUS:" + str(jobstatus))
        # print ("****************************")
        if ( (str(playbookjobstatus['name'])) == playbookjob):    
            results = str(playbookjobstatus['results'])
            jobstatus = str(playbookjobstatus['status'])
            print ("STATUS:" + str(jobstatus))
            print ("RESULTS:" + results)
            zipfile_regex = re.compile('(/home/admin/Config_Backups/(.*)/(.*\.zip))')
            fileparts = re.findall(zipfile_regex, results)
            print ("*************PARTS IS PARTS**************************")
            print (fileparts)
            print ("*************PARTS IS PARTS**************************")
            filename = str(fileparts[0][2])
            downloadpath = "/home/admin/Config_Backups/" + custombackup_tag + "/" + filename
            print(downloadpath)
            myfolder = "./Config_Backups/"
            #savepath =  "/Users/ntrieber/Documents/Scripts/Notebooks/Config_Backups/" + filename
            savepath = myfolder + filename
            print(savepath)
            sftp.get(downloadpath, savepath)
            t.close()
            with open(savepath, 'rb') as f:
                col1.download_button('Download Zip', f, file_name=filename)
            col2expander.write(results)
            col2expander.write ('*********************************\n')
            col1.subheader("LAB RESET COMPLETE AND DEVICE WILL AUTO REBOOT FOR CONFIGS TO TAKE AFFECT!")
            col1.subheader("SOME DEVICES MAY STILL BE REBOOTING!")
            col1.subheader("LAB IS NOW READY FOR USE! (AFTER REBOOTS COMPLETE")
            col2expander.write ('*********************************\n')

        
def goldenlabbackup(col2expander, username, password, loggedin, pica8apikey, authpayload, tokenheaders, pica8apiurl, sgselector, allchecked, deviceschosen, switchgroup_devices, custombackup_tag, datetimestamp, sftp, deviceslist):
    col2expander.write ("Resetting Switches.....Please Standby.....")
    switchListing = {}
    configListing = {}
    if (deviceschosen):
        sgselector = []
        for device in deviceschosen:
            jobspayload=""
            devicebackupurl = pica8apiurl + "/backup_config/" + device
            print (devicebackupurl)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse)
    if (sgselector):
        print ("SG SELECTED!")
        print("**************************")
        print (sgselector)
        print("**************************")
        deviceschosen = []
        sgdevices = []
        for switchgroupname in sgselector:
            print ("Backing up each device in: " + switchgroupname)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("GET", groupswitchlisturl, headers=tokenheaders, data=authpayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse)
            switchgroupslist = json.loads(switchconfigsresponse)
            for switchgroup in switchgroupslist:
                #print (switchgroup)
                #print (switchgroup['name'])
                #switch_groups.append(str(switchgroup['name']))
                if switchgroupname ==  str((switchgroup['name'])):
                    sgdevices = ((switchgroup['sn']))
                    print (sgdevices)
            for device in sgdevices:
                jobspayload=""
                print ("Backing up Device: " + device)
                devicebackupurl = pica8apiurl + "/backup_config/" + device
                print (devicebackupurl)
                keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
                pica8apikey = keyresponse.text
                tokenheaders = settokenheaders(pica8apikey)
                response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
                switchconfigsresponse = response.text
                print (switchconfigsresponse)
    if (allchecked == True):
        deviceschosen = []
        sgselector = []
        alldevices = []
        for device in deviceslist:
            chosendevice = device
            chosendevicestr = str(chosendevice)
            print('Looking up SN For: ' + chosendevicestr)
            print('********************************************\n')
            devserialnum = str(deviceslist.get(chosendevicestr))
            alldevices.append(devserialnum)
        for device in alldevices:
            jobspayload=""
            devicebackupurl = pica8apiurl + "/backup_config/" + device
            print (devicebackupurl)
            keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
            pica8apikey = keyresponse.text
            tokenheaders = settokenheaders(pica8apikey)
            response = requests.request("POST", devicebackupurl, headers=tokenheaders, data=jobspayload, verify=False)
            switchconfigsresponse = response.text
            print (switchconfigsresponse) 
    print ("Lab reset headers: " + str(tokenheaders))
    print ("The key: " + str(pica8apikey))
    playbookname = "CREATE_GOLDEN_CONFIG"
    playbookfile = "playbook.yml"
    playbookfolder = playbookname + "/" + playbookfile
    payload = json.dumps({
        "playbook_name": playbookname,
        "playbook_dir": playbookfolder,
        "switches": deviceschosen,
        "switch_checkall": allchecked,
        "group_list": sgselector,
        "vars": {"custombackup_tag" : custombackup_tag},
        "scheduled": {
            "type": "DIRECT",
            "params": {}
        }
    })
    print (payload)
    # payload = json.dumps({
    #     "config_name": configtodeploy,
    #     "switch_checked": {
    #         "checkall": False,
    #         "checkedNodes": [
    #         {
    #             "sn": switchtodeployserial
    #         }
    #         ],
    #         "uncheckedNodes": []
    #     },
    #     "group_checked": {
    #         "checkall": False,d
    #         "checkedNodes": [],
    #         "uncheckedNodes": []
    #     }
    #     })
    # keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    # pica8apikey = keyresponse.text
    # tokenheaders = settokenheaders(pica8apikey)
    # response = requests.request("POST", deploymenturl, headers=headers, data=payload, verify=False)
    # print(response.text)
    ampconplaybookurl = pica8apiurl + "/ansible/playbooks/run"
    print (ampconplaybookurl)
    keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    pica8apikey = keyresponse.text
    tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("POST", ampconplaybookurl, headers=tokenheaders, data=payload, verify=False)
    switchconfigsresponse = response.text
    print (switchconfigsresponse)
    playbookstatus = json.loads(switchconfigsresponse)
    col2expander.write('Backing up Lab into Golden Demo Configs...Standby')
    # for SwitchConfig in switchconfigs:
    print ('*********************************\n')
    col2expander.write ('*********************************\n')
    # playbookinfo = str(playbookstatus['info'])
    # print("Info:" + playbookinfo)
    # col2expander.write("Info:" + playbookinfo)
    playbookjob = str(playbookstatus['job_name'])
    print("Job:" + playbookjob)
    col2expander.write("Job:" + playbookjob)
    playbookstats = str(playbookstatus['status'])
    print("Status:" + playbookstats)
    col2expander.write("Status:" + playbookstats)
    print ('*********************************\n')
    col2expander.write ('*********************************\n')
    ampconplaybookjobsurl = pica8apiurl + "/ansible/jobs"
    print (ampconplaybookjobsurl)
    jobspayload=""
    jobstatus="IDLE"
    while (jobstatus == "IDLE") or (jobstatus == "RUNNING"):
        keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
        pica8apikey = keyresponse.text
        tokenheaders = settokenheaders(pica8apikey)
        response = requests.request("GET", ampconplaybookjobsurl, headers=tokenheaders, data=jobspayload, verify=False)
        switchconfigsresponse = response.text
        #print (switchconfigsresponse)
        playbookjobsstatus = json.loads(switchconfigsresponse)
        for playbookjobstatus in playbookjobsstatus:
            # print ("ELEMENT: ")
            # print ("****************************")
            # print (playbookjobstatus)
            # print ("****************************")
            # print ("STATUS:" + str(jobstatus))
            # print ("****************************")
            if ( (str(playbookjobstatus['name'])) == playbookjob):
                jobstatus = str(playbookjobstatus['status'])
                print("Job:" + playbookjob)
                print ("Found:" + str(playbookjobstatus['name']))
                print ("STATUS:" + str(jobstatus))
                col2expander.write("Waiting for Backups to Complete......Standby.....")
    keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    pica8apikey = keyresponse.text
    tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("GET", ampconplaybookjobsurl, headers=tokenheaders, data=jobspayload, verify=False)
    switchconfigsresponse = response.text
    #print (switchconfigsresponse)
    playbookjobsstatus = json.loads(switchconfigsresponse)
    for playbookjobstatus in playbookjobsstatus:
        # print ("ELEMENT: ")
        # print ("****************************")
        # print (playbookjobstatus)
        # print ("****************************")
        # print ("STATUS:" + str(jobstatus))
        # print ("****************************")
        if ( (str(playbookjobstatus['name'])) == playbookjob):    
            results = str(playbookjobstatus['results'])
            jobstatus = str(playbookjobstatus['status'])
            print ("STATUS:" + str(jobstatus))
            print ("RESULTS:" + results)
            col2expander.write(results)
            col2expander.write ('*********************************\n')
            col1.subheader("LAB GOLDEN CONFIG CREATION COMPLETE!")
    



#for percent_complete in range(100):
     #my_bar.progress(percent_complete + 1)
username, password, loggedin, pica8apikey, authpayload, runlabresetbutton, backupthelabbutton, goldenlabbackupbutton  = LOGIN(username, password, loggedin, pica8apikey, authpayload, runlabresetbutton, backupthelabbutton, goldenlabbackupbutton)
print (loggedin)
     
if (loggedin == "success"):
    #pica8apikey = pica8apikey    
    tokenheaders = {
        'Accept': 'application/json',
        'Authorization': 'Bearer {}'.format(pica8apikey), 
        'Content-Type': 'application/json',
    }
    print ("The key: " + str(pica8apikey))
    print ('-------------------------------------------------')
    print ("Key Before: " + pica8apikey)
    print ('-------------------------------------------------')
    
    switchlisturl = pica8apiurl + "/switch/all_switch_list"
    # keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    # pica8apikey = keyresponse.text
    
    # print ('-------------------------------------------------')
    # print ("Key After: " + pica8apikey)
    # print ('-------------------------------------------------')
    
    # tokenheaders = settokenheaders(pica8apikey)
    
    print ('-------------------------------------------------')
    print ("Headers: " + str(tokenheaders))
    print ('-------------------------------------------------')
    # keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    # pica8apikey = keyresponse.text
    # tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("GET", switchlisturl, headers=tokenheaders, data=authpayload, verify=False)
    switchconfigsresponse = response.text
    print ('----------------------TEST-----------------------')
    print ('-------------------------------------------------')
    print (switchconfigsresponse)
    print ('------------------------------------------------')
    print ('---------------------TEST-----------------------')
    print ('------------------------------------------------')
    
    print ('-------------------------------------------------')
    print ("Key Before Groups: " + pica8apikey)
    print ('-------------------------------------------------')
    # keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    # pica8apikey = keyresponse.text
    # tokenheaders = settokenheaders(pica8apikey)
    print ('-------------------------------------------------')
    print ("Key After RE-DO: " + pica8apikey)
    print ('-------------------------------------------------')
    groupswitchlisturl = pica8apiurl + "/switch/groups"
    switch_groups = []
    switchgroup_devices = []
    deviceslist = {}
    devicenames = []
    deviceschosen = []
    playbooks = []
    authpayload = ''
    print('YOOOOYOOOOO!')
    # tokenheaders = {
    #     'Accept': 'application/json',
    #     'Authorization': 'Bearer {}'.format(pica8apikey), 
    #     'Content-Type': 'application/json',
    # }
    
    print ("The key NOW: " + str(pica8apikey))
    print ("Group headers: " + str(tokenheaders))
    # keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    # pica8apikey = keyresponse.text
    # tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("GET", groupswitchlisturl, headers=tokenheaders, data=authpayload, verify=False)
    print (response)
    switchconfigsresponse = response.text
    print (switchconfigsresponse)
    print ("PRE CRASH: " + switchconfigsresponse + "\n" + "THE KEY CRASH: " + pica8apikey)
    try:
        switchgroupslist = json.loads(switchconfigsresponse)
        print ('This be the groups: ' + str(switchgroupslist))
    except:
        labmanager_restart_command = "systemctl restart pica8labmanager"
        subprocess.call(['bash', '-c', labmanager_restart_command])
    #switchgroupslist = json.loads(switchconfigsresponse)
    for switchgroup in switchgroupslist:
        #print (switchgroup)
        #print (switchgroup['name'])
        switch_groups.append(str(switchgroup['name']))
        #switchgroup_devices.append(str(switchgroup['sn']))
    print ("The key NOW: " + str(pica8apikey))
    print ("Group headers: " + str(tokenheaders))

    switchlisturl = pica8apiurl + "/switch/all_switch_list"
    # keyresponse = requests.request("POST", tokenurl, headers=tokenheaders, data=authpayload, verify=False)
    # pica8apikey = keyresponse.text
    # tokenheaders = settokenheaders(pica8apikey)
    response = requests.request("GET", switchlisturl, headers=tokenheaders, data=authpayload, verify=False)
    switchconfigsresponse = response.text
    print (switchconfigsresponse)
    switcheslist = json.loads(switchconfigsresponse)
    for switch in switcheslist:
        #print (switchgroup)
        #print (switchgroup['name'])
        devicename=(str(switch['host_name']))
        devicenames.append(devicename)
        devicesn=(str(switch['sn']))
        deviceslist.update({devicename: devicesn})
    
    col1.markdown(f"# STEP 1: Select a Group of Devices, or Individual Devices")


    alldevicegroupschosen = col1.empty()
    alldeviceschosen = col1.empty()
    datechosen = col1.empty()
    #playbookschosen = col1.empty()

    alldevices = col1.checkbox("Select This Option to Backup/SET/RESET ALL Devices in AmpCon", disabled=False, key="1")
    #goldendemoselector = playbookschosen.multiselect   ("Select a Group or Multiple Groups of Devices",playbooks, key="goldendemoselector", disabled=False)
    sgselector = alldevicegroupschosen.multiselect("Select a Group or Multiple Groups of Devices",switch_groups, key="sgselector", disabled=False)
    dvselector = alldeviceschosen.multiselect("Select a Device or Devices",devicenames, key="dvselector", disabled=False)

    
    if alldevices:
        alldevicegroupschosen.multiselect("Select a Group or Multiple Groups of Devices",switch_groups, key="sgselector1", disabled=True)
        alldeviceschosen.multiselect("Select a Device or Devices",devicenames, key="dvselector1", disabled=True)
        allchecked = True 
        col1.markdown(f"# Currently Selected: ALL DEPLOYED DEVICES")
    else:
        allchecked = False   

    if dvselector:
        for device in dvselector:
            chosendevice = device
            chosendevicestr = str(chosendevice)
            print('Looking up SN For: ' + chosendevicestr)
            print('********************************************\n')
            devserialnum = str(deviceslist.get(chosendevicestr))
            deviceschosen.append(devserialnum)
        col1.markdown(f"# Currently Selected {dvselector}")
        print (deviceschosen)

    if sgselector:
        print ("The key: " + str(pica8apikey))
        print ("Group headers: " + str(tokenheaders))

        col1.markdown(f"# Currently Selected {sgselector}")
    
    col1.markdown(f"# STEP 2: BACKUP & RESTORE, or LAB RESET")   
    col1.subheader(f"Config Backups *MUST* be done before ANY Lab change over.\n\n")
    col1.write("*********************************\n\n")
    col1.subheader("**NOTE: BACKUP FILES STORAGE**\n\n WILL BE STORED in BOTH 'SET' and 'JSON' RESTORE FORMATS:")
    col1.markdown(
        f"""
        - ON SWITCH/DEVICE 
            - (in /home/admin - for 'load' restore purposes from CLI)\n
        **AND**\n 
        - ON AMPCON 
            - (in /home/admin/Backups/<customtag> Folder)\n
        **AND**\n 
        - 'NATIVELY' through AmpCon's 'Config Backup'\n
            - as an individual 'Snapshot'\n
        **AND**\n     
        - An archive 'Zip' file of all the configs for all devices will be Downloadable by The 'Download Zip File' button that will appear when the backup completes :) \n
        """)                   
    col1.write("*********************************\n\n")
    col1.subheader(f"**OPTIONAL:**\n\n ENTER A CUSTOM TAG BELOW TO BE USED WITH BACKUP (and RESTORE) OF CONFIGS (AND ADDED TO BACKUP CONFIG FILENAME) FOR CUSTOM RESTORE")
    custombackup_tag = col1.text_input('ENTER CUSTOM TAG:')
    col1.subheader("**NOTE: SAVED FILENAMES**\n\n Files will automatically have the hostname and date followed by your custom tag followed by the format-style of the file (I.e. SET commands format, or native Pica8 XML restore format).\n\n Files will be saved both on device (/home/admin/<filename>) and on AmpCon (/home/admin/Config_Backups/<tagname>/) for future load/save):\n\n I.e. EVPN-ACC2_01-01-2022_MyTag_SET/RESTORE.conf")
    col1.subheader("**ONLY IF RUNNING A RESTORE**: \n\n")
    col1.markdown(f"SELECT THE DATE WHEN THE LAB CONFIG BACKUP **WAS MADE**:")
    dateselected = col1.date_input("SELECT DATE WHEN BACKUP **WAS MADE**:")
    col1.markdown(f"ENTER THE CUSTOM TAG **USED TO MAKE THE PREVIOUS BACKUP OF THE LAB**. THIS CONFIGURATION WILL BE RESTORED TO ALL SELECTED DEVICES.")
    customrestore_tag = col1.text_input('ENTER THE CUSTOM TAG FROM THE PREVIOUS BACKUP.')
    col1.write("*********************************\n\n")
    col1.markdown(f"# STEP 3: PRESS A BUTTON :)")
    backupthelablocallybutton = col1.button('BACKUP THE LAB\'s\n \'CURRENT STATE\'')
    #goldenlabbackupbutton = col1.button('MAKE THE LAB\'s CURRENT CONFIG \'THE GOLDEN CONFIG\'')
    col1.subheader("**NOTE: LAB RESETS**\n\n  Lab Configs will be backed up automatically before reset, if ***no*** custom \"backup\" tag is provided **ABOVE** in the **OPTIONAL** custom BACKUP tag field, a custom backup tag will be created automatically in the form of: current_lab_config+current date/time and added to the filename.\n\n")
    runbasicconfigbutton = col1.button('RESET THE LAB TO A VERY BASIC CONFIG')
    runlabresetbutton = col1.button('RESET THE LAB TO EVPN-CENTRALLY-ROUTED DEMO CONFIG')
    runlabrestorebutton = col1.button('RESET THE LAB TO CONFIG SAVED WITH CUSTOM TAG AND DATE')  
    backup_date = str(dateselected)
    #privatekeyfile = os.path.expanduser('/home/ubuntu/.ssh/id_rsa')
    privatekeyfile = os.path.expanduser('/Users/ntrieber/.ssh/id_rsa')
    
    mykey = paramiko.RSAKey.from_private_key_file(privatekeyfile)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    t = paramiko.Transport(pica8instance, 22)
    #transport = ssh.connect('pica8instance', username = 'admin', pkey = mykey)
    t.connect(username = 'admin', pkey = mykey)
    sftp = paramiko.SFTPClient.from_transport(t)
    print (sftp)
    
    # print (backup_date)
    # print (dateselected)
    # print (str(dateselected))
    if ((backupthelablocallybutton and custombackup_tag) and (sgselector or dvselector or alldevices)):
        print("************************************")
        print("STARTING BACKUP!!!!!!!")
        print("************************************")
        with st.spinner("Backing Up CURRENT Configs natively in AmpCon and to the /home/admin on AmpCon and on the Devices themselves! "):
                        col2expander = col1.expander("Backup Progress Viewer (Press the '+' to Monitor Progress) ")
                        with col2expander:
                            backupthelablocally(col2expander, username, password, loggedin, pica8apikey, authpayload, tokenheaders, pica8apiurl, sgselector, allchecked, deviceschosen, switchgroup_devices, custombackup_tag, datetimestamp, sftp, deviceslist)
                
    # if goldenlabbackupbutton:
    #     print("************************************")
    #     print("Copying CURRENT LAB CONFIGS TO GOLDEN CONFIGS!!!!!!!")
    #     print("************************************")
    #     with st.spinner("Copying CURRENT LAB CONFIGS TO GOLDEN CONFIGS!!!!!!!"):
    #                     col2expander = col1.expander("Backup Progress Viewer (Press the '+' to Monitor Progress) ")
    #                     with col2expander:
    #                         goldenlabbackup(col2expander, username, password, loggedin, pica8apikey, authpayload, tokenheaders, pica8apiurl, sgselector, allchecked, deviceschosen, switchgroup_devices, custombackup_tag, datetimestamp, sftp, deviceslist)
                
    if runlabresetbutton:
        print("************************************")
        print("STARTING LAB RESET TO GOLDEN DEMO CONFIGS!!!!!!!")
        print("************************************")
        with st.spinner("Resetting LAB Devices to Centrally Routed EVPN DEMO Config, Standby..."):
                        col2expander = col1.expander("LAB RESET Progress Viewer (Press the '+' to Monitor Progress) ")
                        with col2expander:
                            labreset(col2expander, username, password, loggedin, pica8apikey, authpayload, tokenheaders, pica8apiurl, sgselector, allchecked, deviceschosen, switchgroup_devices, custombackup_tag, datetimestamp, sftp, deviceslist)
    
    if runbasicconfigbutton:
        print("************************************")
        print("STARTING LAB RESET TO BASIC CONFIGS!!!!!!!")
        print("************************************")
        with st.spinner("Resetting LAB Devices to BASIC Config, Standby..."):
                        col2expander = col1.expander("LAB RESET Progress Viewer (Press the '+' to Monitor Progress) ")
                        with col2expander:
                           labbasicconfigreset(col2expander, username, password, loggedin, pica8apikey, authpayload, tokenheaders, pica8apiurl, sgselector, allchecked, deviceschosen, switchgroup_devices, custombackup_tag, datetimestamp, sftp, deviceslist)
       
                
    if (( runlabrestorebutton and customrestore_tag) and (sgselector or dvselector or alldevices)):
        print("************************************")
        print("STARTING LAB RESET TO " + str(customrestore_tag) + " CONFIGS!!!!!!!")
        print("************************************")
        with st.spinner("Resetting LAB Devices to " + str(customrestore_tag) + " Config Standby..."):
                        col2expander = col1.expander("LAB RESET Progress Viewer (Press the '+' to Monitor Progress) ")
                        with col2expander:
                            labrestore(col2expander, username, password, loggedin, pica8apikey, authpayload, tokenheaders, pica8apiurl, sgselector, allchecked, deviceschosen, switchgroup_devices, custombackup_tag, customrestore_tag, dateselected, datetimestamp, sftp, deviceslist)
                
    
    
    