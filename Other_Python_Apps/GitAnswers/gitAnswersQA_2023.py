from github import Github
import git
import re
import html2text
import requests
import sys
import os, shutil
import os.path
import sqlite3
import sqlalchemy
import glob
import tabulate
import io
from lxml import etree
from pathlib import Path
import boto3
import botocore
import types
import urllib
import subprocess
import json
import simplejson
import datetime
from urllib.parse import urlparse
import scrapy
import re
import hashlib
import os
from annotated_text import annotation
# from json import JSONDecodeError
import logging
from markdown import markdown
import markdown
import streamlit
import streamlit as st
from utils.haystack import query 
from utils.haystack import query as askquestion
from utils.ui import reset_results, set_initial_state
import json
import os
import sys

import pandas as pd
# import streamlit as st
from annotated_text import annotated_text
#from streamlit.legacy_caching.hashing import _CodeHasher
import collections
import functools
import inspect
import textwrap
from streamlit.web.server import Server
# from streamlit.scriptrunner.script_run_context
import time
import random
import string
answer = ''
text = ''
context = ''
image = "githubleaks.jpeg"
caption = "RE-SEARCH. GIT Answers."
sidebarimage = "githubleaks.jpeg"
loggedin = ''
loginresult = ''
st.set_page_config(layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',
              (username, password))
    conn.commit()


def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',
              (username, password))
    data = c.fetchall()
    return data


def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data


def annotate_answer(answer, context):
    start_idx = context.find(answer)
    end_idx = start_idx+len(answer)
    annotated_text(context[:start_idx],
                   (answer, "ANSWER", "#8ef"), context[end_idx:])

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


def fancy_cache(func=None, ttl=None, unique_to_session=False, **cache_kwargs):
    """A fancier cache decorator which allows items to expire after a certain time
    as well as promises the cache values are unique to each session.
    Parameters
    ----------
    func : Callable
        If not None, the function to be cached.
    ttl : Optional[int]
        If not None, specifies the maximum number of seconds that this item will
        remain in the cache.
    unique_to_session : boolean
        If so, then hash values are unique to that session. Otherwise, use the default
        behavior which is to make the cache global across sessions.
    **cache_kwargs
        You can pass any other arguments which you might to @st.cache
    """
    # Support passing the params via function decorator, e.g.
    # @fancy_cache(ttl=10)
    if func is None:
        return lambda f: fancy_cache(
            func=f,
            ttl=ttl,
            unique_to_session=unique_to_session,
            **cache_kwargs
        )

    # This will behave like func by adds two dummy variables.
    dummy_func = st.cache(
        func=lambda ttl_token, session_token, *func_args, **func_kwargs:
        func(*func_args, **func_kwargs),
        **cache_kwargs)

    # This will behave like func but with fancy caching.
    @functools.wraps(func)
    def fancy_cached_func(*func_args, **func_kwargs):
        # Create a token which changes every ttl seconds.
        ttl_token = None
        if ttl is not None:
            ttl_token = int(time.time() / ttl)

        # Create a token which is unique to each session.
        session_token = None
        if unique_to_session:
            session_token = get_session_id()

        # Call the dummy func
        return dummy_func(ttl_token, session_token, *func_args, **func_kwargs)
    return fancy_cached_func


def fancy_cache_demo():
    """Shows how to use the @fancy_cache decorator."""

    st.write('## ttl example')

    @fancy_cache(ttl=1)
    def get_current_time():
        return time.time()
    for i in range(10):
        st.write("This number should change once a second: `%s` (iter: `%i`)" %
                 (get_current_time(), i))
        time.sleep(0.2)

    st.write('## unique_to_session example')

    @fancy_cache(unique_to_session=True)
    def random_string(string_len):
        return ''.join(random.sample(string.ascii_lowercase, string_len))
    for i in range(3):
        st.write("This string shouldn't change, but should differ by session: `%s` (iter: `%i`)" %
                 (random_string(10), i))


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



local_css("gitAnswers.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = ''
image = "githubleaks.jpeg"
caption = "RE-SEARCH. RE-DISCOVERED"
sidebarimage = "githubleaks.jpeg"







# Create ensembled pipeline


set_initial_state()
# col1, col2 = st.columns(2)
st.sidebar.image(image, caption=caption, width=250, use_column_width=False,
                 clamp=False, channels='RGB', output_format='auto')
st.sidebar.title("GitAnswers Q&A Portal\n GitHub Researcher")
username = st.sidebar.text_input('username')
password = st.sidebar.text_input('password', type='password')

#state.logintogitlit = st.sidebar.button("LOGIN")
# DB Management
conn = sqlite3.connect('streamuser.db')
c = conn.cursor()
# DB  Functions
# @fancy_cache(unique_to_session=True)


@cache_on_sidebarbutton_press('LOGIN')
def authenticate(username, password, loggedin):
    hashed_pswd = make_hashes(password)
    result = login_user(username, check_hashes(password, hashed_pswd))
    if result:
        loggedin = 1
    elif not (result):
        st.sidebar.error("Invalid Login. Please try again.")
    return loggedin


if authenticate(username, password, loggedin):
    st.sidebar.success("Logged In as {}".format(username))
    changepassexpander = st.sidebar.expander("Change Password")
    with changepassexpander:
        currentpass = password
        newpass = changepassexpander.text_input(
            'New Password', value='Enter New Password', type='password')
        newpassconfirm = changepassexpander.text_input(
            'Confirm New Password', value="Confirm NEW Password", type='password')
        changepass = changepassexpander.button("CHANGE PASSWORD")
        hashed_pswd = make_hashes(newpassconfirm)
        if (newpass == currentpass):
            changepassexpander.error(
                "New Password same as current! Passwords MUST be DIFFERENT!")
        elif ((newpass != currentpass) and (newpass == newpassconfirm)):
            #c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, currentpass))
            c.execute('UPDATE userstable SET password=? WHERE username=?',
                      (hashed_pswd, username))
            conn.commit()
            st.success("PASSWORD CHANGED SUCCESSFULLY!")
            #st.success("You may have to login again!")
            caching.clear_cache()
            #cursor.execute('UPDATE userInfo SET username=? WHERE username=?', (new_username, old_username))
        elif ((newpass != currentpass) and (newpass != newpassconfirm)):
            changepassexpander.error("Please enter new password and confirm.")

    # UI search bar and sidebar
    # st.sidebar.image(image, caption=caption, width=250, use_column_width=False,
    #                 clamp=False, channels='RGB', output_format='auto')
    # st.sidebar.title("GitAnswers")
    st.image(image, caption=None, width=None, use_column_width=True,
                clamp=False, channels='RGB', output_format='auto')
    st.write("# GitAnswers Q/A Demo - Readmes and Docs from Repos")

    # Search bar
    question = st.text_input("Ask A Question", value=st.session_state.question, max_chars=1000, on_change=reset_results)

    run_pressed = st.button("Run")

    run_query = (
        run_pressed or question != st.session_state.question
    )


    # Get results for query
    if run_query and question:
        reset_results()
        st.session_state.question = question
        with st.spinner("🔎 &nbsp;&nbsp; Running your pipeline"):
            try:
                st.session_state.results = askquestion(question)
            except JSONDecodeError as je:
                st.error(
                    "👓 &nbsp;&nbsp; JSON DECODER had An error occurred reading the results. Is the document store working?"
                )    
            except Exception as e:
                logging.exception(e)
                st.error("🐞 &nbsp;&nbsp; A general error occurred during the request.")
            
                

    if st.session_state.results:
        st.write('## We have The Answers for You:')
        st.write('*******************************')
        answers = st.session_state.results
        # count = range(len(answers))
        for i in range(len(answers)):
            ananswer = (answers[i])
            # theanswer=json.loads(answerparts)
            # theanswer=[x.to_dict() for x in answerparts]
            print ("HAZZAH IT WORKED! " + str(ananswer))
            for x, y in ananswer.items():
                #print(key, '->', value)
            
                # print (answerparts[j])
                if (x == "answer"):
                    text = y
                if (x == "context"):
                    context = y
                    
                start_idx = context.find(text)
                end_idx = start_idx + len(text)
                print ("***********************************")
                print (str(context.find(text)))
                print (context[:start_idx])
                print (context[end_idx:])
                print (text)
                print ("***********************************")
                st.write ("*****************************")
                st.markdown(context[:start_idx] + str(annotation(body=text, label="ANSWER", background="#964448", color='#ffffff')) + context[end_idx:], unsafe_allow_html=True)
                st.write ("*****************************")           
                st.write(
                    markdown.markdown(context[:start_idx] + str(annotation(body=text, label="ANSWER", background="#964448", color='#ffffff')) + context[end_idx:]),
                    unsafe_allow_html=True,
                )
                #     # else:
                #     #     st.info("🤔 &nbsp;&nbsp; Haystack is unsure whether any of the documents contain an answer to your question. Try to reformulate it!")
