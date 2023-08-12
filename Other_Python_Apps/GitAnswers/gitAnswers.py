# annotate text is from Steamlit's st-annotated-text 1.1.0
#from annotated_text import annotated_text
from github import Github
import git
import re
import html2text
import requests
import sys
import os, shutil
import os.path
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
import os

#from haystack import Finder
#from haystack.preprocessor.cleaning import clean_wiki_text
#from haystack.preprocessor.utils import convert_files_to_docs, fetch_archive_from_http
#from haystack.reader.farm import FARMReader
#from haystack.reader.transformers import TransformersReader
#from haystack.utils import print_answers
#from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
from haystack.utils import convert_files_to_docs, fetch_archive_from_http
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import PreProcessor
from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader
from streamlit.components.v1 import html
from docx import *
from pathlib import Path
from elasticsearch import Elasticsearch
import tika
import uuid
tika.TikaClientOnly = True
from tika import parser
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
import os, pathlib
from subprocess import Popen, PIPE, STDOUT
from scrapy.utils.python import to_native_str
import logging
from scrapy.spiders import CrawlSpider, Rule
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapyscript import Job, Processor
from scrapy.spiders import Spider
from scrapy import Request
from tld import get_tld
from twisted.internet import reactor
#from testspiders.spiders.followall import FollowAllSpider
from scrapy.utils.project import get_project_settings
import datetime
from datetime import datetime as date
import streamlit as st
import altair as alt
import plotly.express as px
import hashlib
import sqlite3
# import SessionState
# from streamlit.caching.hashing import _CacheFuncHasher
# from streamlit.caching.hashing import _CacheFuncHasher as _CodeHasher
#from st import runtime.legacy_caching.clear_cache as legacy_caching
import collections.abc
import functools
import inspect
import textwrap
# import streamlit.report_thread as ReportThread
from streamlit.web.server import Server
import time
import random
import string
# from streamlit.web.server import Server
# from streamlit.scriptrunner.script_run_context
import time
import random
import string
try:
    from streamlit.scriptrunner.script_run_context import get_script_run_ctx as get_report_ctx 
    from streamlit.scriptrunner.script_run_context import add_script_run_ctx
    from streamlit.script_run_context import add_script_run_ctx as add_report_ctx
except ModuleNotFoundError:
    # streamlit < 1.8
    try:
        from streamlit.scriptrunner.script_run_context import get_script_run_ctx  # type: ignore
    except ModuleNotFoundError:
        print ('FOOBAR!')
        # # streamlit < 1.4
        # from streamlit.report_thread import (  # type: ignore
        #     get_script_run_ctx as get_report_ctx
        # )

# try:
#     # Before Streamlit 0.65
#     from streamlit.ReportThread import get_report_ctx
#     from streamlit.web.server import Server
# except ModuleNotFoundError:
#print ("FOOBAR")
# After Streamlit 0.65
# from streamlit.scriptrunner.script_run_context import get_report_ctx
# from streamlit.web.server import Server
# try:
#     # Before Streamlit 0.65
#     from streamlit.ReportThread import get_report_ctx
#     from streamlit.web.server import Server
# except ModuleNotFoundError:
#     # After Streamlit 0.65
#     from streamlit.report_thread import get_report_ctx
#     from streamlit.web.server import Server
#st.cache(persist=True, allow_output_mutation=True)
#from streamlit import caching
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor
from scrapy.item import Item, Field
st.set_page_config(layout="wide")



thedate = date.now().strftime("%m/%d/%Y")
todaysdate = datetime.date.today().isoformat() + "_" + \
    str(datetime.datetime.now().time().isoformat()).replace(':', '-')
username = ''
apikey = ''
wpageauthor = ''
htmltext = ''
docUUID = ''
docURL = ''
docDate = ''
docTitle = ''
docContent = ''
docAuthor = ''
baseurl = ''
s3objects = []
s3bucketname = ''
s3fileobjectname = ''
S3ingestpath = ''
S3bucketname = ''
S3connector = ''
gitanswersVolumeName = ''
gitanswersapikey = ''
gitanswersjobtoadd = ''
volumeDescript = ''
volumesURL = ''
volumeId = ''
volumeid = ''
VolumeID = ''
VolumeStatus = ''
volumesListing = {}
volumeIdslist = {}
volumeName = ''
bearerstr = ''
decruftenabled = ''
deeperdiscovery = ''
gitanswersinstance = ''
isS3folder = 'True'
chosenVolume = ''
chosenVolumestr = ''
gitfolder = ''
urltograb = ''
urltocrawl = ''
alloweddomain = ''
dirname = ''
reponame = ''
repostars = ''
repoforks = ''
repopushdate = ''
repoactive = ''
image = "githubleaks.jpeg"
caption = "RE-SEARCH. GIT Answers."
sidebarimage = "githubleaks.jpeg"
loggedin = ''
loginresult = ''
repolastupdated = ''

ACCESS_TOKEN = ''

g = Github(ACCESS_TOKEN)

gitanswersheaders = {
    'Accept': 'application/json',
    'Authorization': 'Bearer {}'.format(gitanswersapikey),
    'Content-Type': 'application/json',
}


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)


local_css("labman.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')


def open_page(url):
    cmd1 = "streamlit run ./gitAnswersQA_2023.py &"
    subprocess.call(['bash', '-c', cmd1])
    print ("opening: " + str(url))
    time.sleep(10)
    open_script= """
        <script type="text/javascript">
            window.open('%s', '_blank').focus();
        </script>
    """ % (url)
    print ("here's the script: " + str(open_script))
    html(html=open_script)


# This is a default usage of the PreProcessor.
# Here, it performs cleaning of consecutive whitespaces
# and splits a single large document into smaller documents.
# Each document is up to 1000 words long and document breaks cannot fall in the middle of sentences
# Note how the single document passed into the document gets split into 5 smaller documents

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=100,
    split_respect_sentence_boundary=True,
)
#docs_default = preprocessor.process([doc_txt])
#print(f"n_docs_input: 1\nn_docs_output: {len(docs_default)}")

#converter = TextConverter(remove_numeric_tables=True, valid_languages=["en"])
#doc_txt = converter.convert(file_path="data/tutorial8/classics.txt", meta=None)[0]

#converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
#doc_pdf = converter.convert(file_path="data/tutorial8/bert.pdf", meta=None)[0]

#converter = DocxToTextConverter(remove_numeric_tables=False, valid_languages=["en"])
#doc_docx = converter.convert(file_path="data/tutorial8/heavy_metal.docx", meta=None)[0]

# A data storage class(like directory) to store the extracted data


class PageContentItem(Item):
    url = Field()
    content = Field()


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


def save_file(link, dirname, reponame, S3ingestpath, bucket):
    fileurltodownload = link
    #dirname = str(os.path.dirname(readmefilepath))
    filetodownload = urlparse(fileurltodownload)
    # Output: /kyle/09-09-201315-47-571378756077.jpg
    print(str(filetodownload.path))
    filenametodownload = str((os.path.basename(filetodownload.path)))
    print("Saving File from crawler: " + filenametodownload)
    col2expander.write("Saving File from crawler: " + filenametodownload)
    urllib.request.urlretrieve(
        fileurltodownload, dirname + '/' + reponame + '_reference_' + filenametodownload)
    filepath = dirname + '/' + reponame + '_reference_' + filenametodownload
    # new S3 Upload section
    col2expander.write("********************************************")
    print("* UPLOADING Github Research TO gitanswers STORAGE NOW")
    col2expander.write("* UPLOADING Github Research TO gitanswers STORAGE NOW")
    print("********************************************")
    col2expander.write("********************************************")
    print("Uploading File: " + filepath + " to S3!")
    col2expander.write("Uploading File: " + filepath + " to S3!")
    s3fileobjectname = S3ingestpath + '/' + reponame + '_' + filenametodownload
    key = bucket.Object(s3fileobjectname)
    print("********************************************")
    try:
        with open(filepath, 'rb') as data:
            key.upload_fileobj(data)
    except:
        pass
    # new S3 upload section


def list_paths(root_tree, path=Path(".")):
    for blob in root_tree.blobs:
        yield path / blob.name
    for tree in root_tree.trees:
        yield from list_paths(tree, path / tree.name)


def fileInRepo(repo, filePath):
    '''
    repo is a gitPython Repo object
    filePath is the full path to the file from the repository root
    returns true if file is found in the repo at the specified path, false otherwise
    '''
    pathdir = os.path.dirname(filePath)

    # Build up reference to desired repo path
    rsub = repo.head.commit.tree

    for path_element in pathdir.split(os.path.sep):

        # If dir on file path is not in repo, neither is file.
        try:
            rsub = rsub[path_element]

        except KeyError:

            return False

    return(filePath in rsub)


def contributor_counts(contributorslist):
    totalcontributions = 0
    contributors = 0
    print("\n\n")
    print("*****************************************")
    for contributor in contributorslist:
        contributors += 1
        print(str(contributor['login']))
        contributions = (contributor['contributions'])
        totalcontributions += contributions
    print("*****************************************")
    contributors += 1
    return contributors, totalcontributions


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


class gitextender(CrawlSpider):
    #print ("Spidering......Standby......")
    name = 'githubextender'
    allowed_domains = [alloweddomain]
    start_urls = [urltocrawl]
    # callback='parse_item'
    #reponame = "blockbench"
    #dirname = "blockbench"
    #urltocrawl = url

    # allowed_domains.append(alloweddomain)
    # start_urls.append(urltocrawl)
    ext = ['.pdf', '.doc', '.docx', '.txt', '.html', '.rtf']
    rules = (
        # Rule(LxmlLinkExtractor(allow_domains=(alloweddomain), ),
        Rule(
            # process_links=process_links,
            # callback='parse_item',
            follow=False
        ),
    )

    def start_requests(self):
        print("REQUEST Spidering Using These parameters:")
        print("*********************************")
        print("Researching extended info from Repo:" + self.reponame)
        print("Allowed Domain to Spider: " + self.alloweddomain)
        print("Spidering: " + self.urltocrawl)
        print("Saving to: " + self.dirname)
        yield Request(self.urltocrawl, callback=self.parse_item)

    def parse_item(self, response):
        boto3.setup_default_session(profile_name='')
        client = boto3.client("s3")
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(self.S3bucketname)
        print("NEW PARSE ITEM Spidering Using These parameters:")
        print("*********************************")
        print("Researching extended info from Repo:" + self.reponame)
        print("Allowed Domain to Spider: " + self.alloweddomain)
        print("Spidering: " + self.urltocrawl)
        print("Saving to: " + self.dirname)
        print("********SAVING FILE*************************")
        # print(response.text)
        # print(response.body)
        # print("********SAVING*************************")
        print("URL: " + response.url)
        link = response.url
        title = response.xpath('//title/text()').extract()

        if link.endswith(tuple(self.ext)):
            save_file(link, self.dirname, self.reponame,
                      self.S3ingestpath, bucket)
        else:
            pagename = response.url.split("/")[-1] + '.html'
            print("Supposed to save this page: " + pagename)
            filename = self.dirname + '/' + self.reponame + '_reference_' + pagename
            print("opening: " + filename + " to save the page to......")
            print("Saving Data:")
            print("*********************************")
            # print(response.text)
            # print(response.body)
            # print("*********************************")
            with open(filename, 'wb') as f:
                print("*********************************")
                print("Saving HTML FILE: " + filename)
                print("*********************************")
                f.write(response.body)
            # new S3 Upload section
            col2expander.write("********************************************")
            print("* UPLOADING Github Research TO gitanswers STORAGE NOW")
            col2expander.write(
                "* UPLOADING Github Research TO gitanswers STORAGE NOW")
            print("********************************************")
            col2expander.write("********************************************")
            print("Uploading File: " + filename + " to S3!")
            col2expander.write("Uploading File: " + filename + " to S3!")
            s3fileobjectname = self.S3ingestpath + '/' + self.reponame + '_' + pagename
            key = bucket.Object(s3fileobjectname)
            print(
                "********************************************")
            try:
                with open(filename, 'rb') as data:
                    key.upload_fileobj(data)
            except:
                pass
            # new section
        return {
            'url': response.url, 'title': title
        }


def githubtogitanswers(filestodownload, reponame, repourl, repoowner, repostars, repoforks, repopushdate, repolastupdated, searchterms):
    print("Reseaching this Repo: " + reponame)
    col2expander.write("Search terms: " + searchterms)
    col2expander.write("Reseaching this Repo: " + reponame)
    print("Located here: " + repourl)
    col2expander.write("Located here: " + repourl)
    #filestodownload = filestodownload
    print("Looking for: " + filestodownload)
    col2expander.write("Looking for: " + filestodownload)
    print("********************************************")
    #cmd1="git clone -n " + repourl + " --depth 1"
    cmd1 = 'function git_sparse_clone() (   rurl="$1" localdir="$2" && shift 2;    mkdir -p "$localdir";   cd "$localdir";    git init;   git remote add -f origin "$rurl";    git config core.sparseCheckout true;    for i; do     echo "$i" >> .git/info/sparse-checkout;   done;    git pull origin main; ); git_sparse_clone ' + '"' + repourl + '"' + ' "./' + reponame + '" "' + filestodownload + '"'
    cmd1a= 'function git_sparse_clone() (   rurl="$1" localdir="$2" && shift 2;    mkdir -p "$localdir";   cd "$localdir";    git init;   git remote add -f origin "$rurl";    git config core.sparseCheckout true;    for i; do     echo "$i" >> .git/info/sparse-checkout;   done;    git pull origin master; ); git_sparse_clone ' + '"' + repourl + '"' + ' "./' + reponame + '" "' + filestodownload + '"'
    #cmd1 = 'function git_sparse_clone() (   rurl=\"$1\" localdir=\"$2\" && shift 2;    mkdir -p \"$localdir";   cd \"\$localdir\";    git init;   git remote add -f origin \"$rurl\";    git config core.sparseCheckout true;    for i; do     echo \"$i\" >> .git/info/sparse-checkout;   done;    git pull origin main; ); git_sparse_clone ' + '"' + repourl + '"' + ' "./' + reponame + '" "' + filestodownload + '"'
    #cmd1a = 'function git_sparse_clone() (   rurl=\"$1\" localdir=\"$2\" && shift 2;    mkdir -p \"$localdir\";   cd \"\$localdir\";    git init;   git remote add -f origin \"$rurl\";    git config core.sparseCheckout true;    for i; do     echo \"$i\" >> .git/info/sparse-checkout;   done;    git pull origin master; ); git_sparse_clone ' + '"' + repourl + '"' + ' "./' + reponame + '" "' + filestodownload + '"'
    cmd2 = 'git_sparse_clone ' + '"' + repourl + '"' + ' "./' + reponame + '" "' + filestodownload + '"'
    print("Going to run these commands:")
    print(cmd1)
    print(cmd2)
    print("********************************************")
    gitfolder = ''
    #cmd2a="cd " + reponame + ";"
    #cmd2b=" git checkout HEAD README.md; mv README.md " + reponame + "_README.md"

    mkdircmd = 'mkdir ' + repoowner
    chdircmd1 = 'cd ./' + reponame
    chdircmd2 = 'cd ./' + repoowner
    chdircmd3 = 'cd ./' + repoowner + '/' + reponame
    print("going to run these commands:")
    print(chdircmd1)
    print(chdircmd2)
    print(chdircmd3)
    print("***********************************************")
    rmgitcmd = 'rm -rf .git'
    print("To remove git: " + rmgitcmd)
    print("***********************************************")
    combocommand = cmd1 + '; ' + cmd2
    combocommand1a = cmd1a + '; ' + cmd2
    print("Combo command is: " + combocommand)
    
    # subprocess.call(['bash', '-c', cmd1])
    if (os.path.exists(reponame)):
        try:
            os.mkdir('./' + repoowner)
            gitfolder = ('./' + repoowner)
            print("using gitfolder: " + str(gitfolder))
            if not (filestodownload == "README.md"):
                print("We are looing for: " + filestodownload +
                      '....deleting git-config now....')
                subprocess.call(['bash', '-c', chdircmd1 + '; ' + rmgitcmd])
                subprocess.call(['bash', '-c', chdircmd2 + '; ' + rmgitcmd])
                subprocess.call(['bash', '-c', chdircmd3 + '; ' + rmgitcmd])
            subprocess.call(['bash', '-c', chdircmd2 + '; ' + cmd1])
        except:
            gitfolder = ('./' + repoowner)
            print("using gitfolder: " + str(gitfolder))
            if not (filestodownload == "README.md"):
                print("We are looing for: " + filestodownload +
                      '....deleting git-config now....')
                subprocess.call(['bash', '-c', chdircmd1 + '; ' + rmgitcmd])
                subprocess.call(['bash', '-c', chdircmd2 + '; ' + rmgitcmd])
                subprocess.call(['bash', '-c', chdircmd3 + '; ' + rmgitcmd])
            subprocess.call(['bash', '-c', chdircmd2 + '; ' + cmd1])
    else:
        print ("Trying....")
        subprocess.call(['bash', '-c', cmd1a])
        subprocess.call(['bash', '-c', cmd1])
        #subprocess.call(cmd1, shell=True)
        #subprocess.call(['bash', '-c', combocommand1a])
        #subprocess.call(['bash', '-c', combocommand])
        gitfolder = './' + reponame
        print("using gitfolder: " + str(gitfolder))
    print("Before downloading/uploading files I'm going use this folder: " + gitfolder)
    print("***************************************************")
    fileparts = filestodownload.split('.')
    filespec = fileparts[1]
    print('Grabbing all files that end with: ' + filespec)
    col2expander.write('Grabbing all files that end with: ' + filespec)
    #boto3.setup_default_session(profile_name='')
    #client = boto3.client("s3")
    #s3 = boto3.resource('s3')
    #keyitem = ''
    #paginator = client.get_paginator("list_objects_v2")
    #s3objects = []
    #bucket = s3.Bucket(S3bucketname)

    for dirpath, dirnames, filenames in os.walk(gitfolder):
        for filename in [f for f in filenames if f.endswith(filespec)]:
            readmesfolder = "./newreadmes/"
            readmefilepath = os.path.join(dirpath, filename)
            filepathparts = (os.path.dirname(readmefilepath)).split('/')
            #print("PARTS: " + filepathparts[1])
            foldertoreplace = ('./' + str(filepathparts[1]))
            #print("REPLACE THIS: " + foldertoreplace)
            #print(os.path.dirname(readmefilepath).replace('./', '').replace(foldertoreplace, ''))
            #print (os.path.join(dirpath, filename))
            repopath = os.path.basename(os.path.dirname(readmefilepath))
            shutil.copy(readmefilepath, readmesfolder + repopath + '_' + filename + ".txt")

        
        for filename in [f for f in filenames if f.endswith(filespec)]:
            readmefilepath = os.path.join(dirpath, filename)
            filepathparts = (os.path.dirname(readmefilepath)).split('/')
            #print("PARTS: " + filepathparts[1])
            foldertoreplace = ('./' + str(filepathparts[1]))
            #print("REPLACE THIS: " + foldertoreplace)
            print(os.path.dirname(readmefilepath).replace('./', '').replace(foldertoreplace, ''))
            print (os.path.join(dirpath, filename))
            repopath = os.path.basename(os.path.dirname(readmefilepath))
            # shutil.copy(readmefilepath, readmesfolder + repopath + '_' + filename)
            if (filespec != 'pdf'):
                print (str(readmefilepath))
                with open(readmefilepath, 'r') as file:
                    # savedreadmepath = "./newreadmes/" + readmefilename
                    # shutil.copy(readmefilepath, savedreadmepath)
                    readmefiledata = file.read().replace('\n', '')
                    #readmefiledata = file.read()
                    docDate = repolastupdated
                    docPath = readmefilepath
                    docTitle = os.path.basename(readmefilepath)
                    docContent = readmefiledata
                    docAuthor = repoowner
                    docStars = repostars
                    docForks = repoforks
                    docRepo = reponame
                    docUUID = docTitle.replace(' ', '_') + "_" + todaysdate
                    docURL = repourl + "/master" + \
                        readmefilepath.replace(foldertoreplace, '')
                    #docDate = dateutil.parser.parse(last_updated)
                    jsondata = dict()
                    # relative filepath
                    jsondata["docUUID"] = docUUID
                    # s3 filepath
                    jsondata["docURL"] = docURL
                    # filename without extension
                    jsondata["docTitle"] = docTitle
                    # current date/time (epochMs - converted)
                    jsondata["docDate"] = docDate
                    # John Smith + random number
                    jsondata["docAuthor"] = docAuthor
                    jsondata["docStars"] = str(docStars)
                    jsondata["docForks"] = str(docForks)
                    jsondata["docPath"] = str(docPath)
                    jsondata["docRepo"] = str(docRepo)
                    #data["docContent"] = docContent.replace(':','\\:')
                    jsondata["docContent"] = docContent
                    print('**************DATA DICT*********************')
                    print(jsondata)
                    print('**************DATA DICT*********************')
                    urlsfile = "./" + reponame + "_urllist.txt"
                    masterurlsfile = "./" + searchterms + "_masterurlsfile.txt"
                    readmeurl = docURL.replace(
                        "github.com", "raw.githubusercontent.com")
                    readmeurlsfile = "./" + searchterms + "_githubreadmeurls.txt"
                    with open(readmeurlsfile, 'a') as f:
                        print("*********************************")
                        print("README Located at: " + readmeurl)
                        print("Saving Readme URL to File: " + readmeurlsfile)
                        print("*********************************")
                        f.write(readmeurl + "\n")
                    with open(masterurlsfile, 'a') as f:
                        print("*********************************")
                        print("README Located at: " + readmeurl)
                        print("Saving Readme URL to File: " + masterurlsfile)
                        print("*********************************")
                        f.write(readmeurl + "\n")
                    #fullpath = os.path.dirname(readmefilepath).replace('./', '')
                    #foldername = os.path.relpath(filepath)
                    readmefilename = (os.path.relpath(
                        readmefilepath)).replace('/', '_') + ".json"
                    jsonfilepath = "./json/" + readmefilename
                    gitanswersjsonfile = open(jsonfilepath, 'a')
                    json.dump(jsondata, gitanswersjsonfile)
            link_regex = re.compile(
                '((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
            extension_regex = re.compile('\.pdf')
            print("Searching: " + readmefilepath + " for URLs...standby....")
            col2expander.write(
                "Searching: " + readmefilepath + " for URLs...standby....")
            with open(readmefilepath) as file:
                try:
                    for line in file:
                        #urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line)
                        urls = re.findall(link_regex, line)
                        # print(urls)
                        for url in urls:
                            urltograb = re.sub('\).*', '', str(url[0]))
                            print("******************************************")
                            col2expander.write(
                                "******************************************")
                            print("Found this URL in the README............")
                            col2expander.write(
                                "Found this URL in the README............")
                            print(urltograb)
                            #urlsfile = "./" + reponame + "_urllist.txt"
                            with open(urlsfile, 'a') as f:
                                print("*********************************")
                                print("Saving URL to File: " + urlsfile)
                                print("*********************************")
                                f.write(urltograb + "\n")
                            with open(masterurlsfile, 'a') as f:
                                print("*********************************")
                                print("Saving URL to File: " + masterurlsfile)
                                print("*********************************")
                                f.write(urltograb + "\n")
                            col2expander.write(urltograb)
                            print("******************************************")
                            col2expander.write(
                                "******************************************")
                            print("Downloading any Discovered PDFs...........")
                            col2expander.write(
                                "Downloading any Discovered PDFs...........")
                            print("******************************************")
                            col2expander.write(
                                "******************************************")
                            pdfmatch = re.match(".*\.pdf", urltograb)
                            print("PDF Found: " + str(pdfmatch))
                            print("******************************************")
                            if not (pdfmatch):
                                dirname = str(os.path.dirname(readmefilepath))
                                urldomain = get_tld(urltograb, as_object=True)
                                alloweddomain = str(urldomain.fld)
                                # alloweddomain=str((get_tld(urltograb, as_object=True))
                                testdomain = str(urldomain.tld)
                                print("Wasn't a match....so.....Try here: " +
                                      testdomain + " Or here: " + alloweddomain)
                                print("******************************************")
                                col2expander.write(
                                    "******************************************")
                                print(
                                    "Only allowing this domain to be spidered: " + alloweddomain)
                                col2expander.write(
                                    "Only allowing this domain to be spidered: " + alloweddomain)
                                print(
                                    "Will soon spider out and download from this URL: " + urltograb)
                                col2expander.write(
                                    "Will soon spider out and download from this URL: " + urltograb)
                                print("******************************************")
                                col2expander.write(
                                    "******************************************")
                                githubJob = Job(gitextender, urltocrawl=urltograb, url=urltograb, reponame=reponame, dirname=dirname,
                                                alloweddomain=alloweddomain, S3bucketname=S3bucketname, S3ingestpath=S3ingestpath)
                                #pythonJob = Job(gitextender, url='http://www.python.org')

                                # Create a Processor, optionally passing in a Scrapy Settings object.
                                processor = Processor(settings=None)

                                # Start the reactor, and block until all spiders complete.
                                #data = processor.run([githubJob, pythonJob])
                                spiderdata = processor.run([githubJob])

                                # Print the consolidated results
                                print("Results from the Spider:")
                                print("***********************************")
                                print(json.dumps(spiderdata, indent=4))
                                #process = CrawlSpider()
                                # process.crawl(githubextender)
                                # process.start() # the script will
                            if (pdfmatch):
                                print("MATCHED!!!")
                                dirname = str(os.path.dirname(readmefilepath))
                                filetodownload = urlparse(urltograb)
                                # Output: /kyle/09-09-201315-47-571378756077.jpg
                                print(str(filetodownload.path))
                                filenametodownload = str(
                                    (os.path.basename(filetodownload.path)))
                                # Output: 09-09-201315-47-571378756077.jpg
                                print(os.path.basename(filetodownload.path))
                                try:
                                    filedownloadpath = str(
                                        dirname + '/' + filenametodownload)
                                    print("Downloading file: " +
                                          filedownloadpath)
                                    col2expander.write(
                                        "Downloading file: " + filedownloadpath)
                                    urllib.request.urlretrieve(
                                        urltograb, dirname + '/' + filenametodownload)
                                    col2expander.write(
                                        "********************************************")
                                    print(
                                        "* UPLOADING Github Research TO gitanswers STORAGE NOW")
                                    col2expander.write(
                                        "* UPLOADING Github Research TO gitanswers STORAGE NOW")
                                    print(
                                        "********************************************")
                                    col2expander.write(
                                        "********************************************")
                                    print("Uploading File: " +
                                          filedownloadpath + " to S3!")
                                    col2expander.write(
                                        "Uploading File: " + filedownloadpath + " to S3!")
                                    s3fileobjectname = S3ingestpath + '/' + reponame + '_' + filenametodownload
                                    key = bucket.Object(s3fileobjectname)
                                    print(
                                        "********************************************")
                                    try:
                                        with open(filedownloadpath, 'rb') as data:
                                            key.upload_fileobj(data)
                                    except:
                                        pass
                                except:
                                    continue

                except:
                    # Add routine to catch "chinese/non-UTF8 in text/README files"
                    continue
            #fullpath = os.path.dirname(readmefilepath).replace('./', '')
            #foldername = os.path.relpath(filepath)
            #readmefilepath = "./" + reponame + "/" + readmefilename
            #s3fileobjectname = S3ingestpath + '/' + readmefilename
            #key = bucket.Object(s3fileobjectname)
            print("********************************************")
            col2expander.write("********************************************")
            print("* UPLOADING Github Readmes Pages TO gitanswers NOW")
            col2expander.write("* Downloads complete Login to GitAnswers Q/A Portal to start researching!")
            print("********************************************")
            col2expander.write("********************************************")
            # doc_dir = "./newreadmes/"
            # dicts = convert_files_to_docs(dir_path=doc_dir, clean_func= processor = PreProcessor(
            #     clean_empty_lines=True,
            #     clean_whitespace=True,
            #     clean_header_footer=True,
            #     split_by="word",
            #     split_length=200,
            #     split_respect_sentence_boundary=True,
            #     split_overlap=0
            #     ))
            # print(dicts[:3])
            # # from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
            # # document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="pica8docs")
            # from haystack.document_stores import InMemoryDocumentStore
            # from haystack.document_stores import ElasticsearchDocumentStore, FAISSDocumentStore
            # # converter = PDFToTextConverter(remove_numeric_tables=True)
            # # #doc_pdf = converter.convert(file_path="data/preprocessing_tutorial/bert.pdf", meta=None)
            # # doc = converter.convert(file_path=filename, meta={'name':str(filename)})
            # # processor = PreProcessor(
            # #     clean_empty_lines=True,
            # #     clean_whitespace=True,
            # #     clean_header_footer=True,
            # #     split_by="word",
            # #     split_length=200,
            # #     split_respect_sentence_boundary=True,
            # #     split_overlap=0
            # #     )
            # # docs = processor.process(doc)
            # # print (docs)
            # # document_store.write_documents(docs)
            # document_store.write_documents(dicts)
            # document_store.save(index_path="./faissshift.index", config_path="./faiss.json")
            # document_store.save("my_faiss")
            # # doc_dir = "/home/ntrieber/Documents/githubresearcher_new/researcher/newreadmes/"
            # # dicts = convert_files_to_docs(dir_path=doc_dir)
            # # print(dicts[:3])
            # # from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
            # # document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="pica8docs")
            # # document_store.write_documents(dicts)
            # # #try:
            # #     #with open(readmefilepath, 'rb') as data:
            # #         #key.upload_fileobj(data)
            # # #except:
            # #     #continue


def search_github(keywords, repopushdate, repomoddate, repostars, repoforks, repoactive, repocontributors, col2expander, gitanswersinstance, s3bucketname, S3ingestpath, gitanswersapikey, S3connector, gitanswersVolumeName, volumeName, repolastupdated):
    #query = '+'.join(keywords) + f' in:readme in:description created:>' + str(repopushdate) + ' stars:>=' + str(repostars) + ' forks:>=' + str(repoforks) + ' archived:' + str(repoactive) + ' pushed:>' + str(repomoddate)
    separator = '_'
    searchterms = separator.join(keywords)
    query = '+'.join(keywords) + f' created:>' + str(repopushdate)
    # adding mirror:true could severely limit results by only showing repos with a dependency / that are mirrored
    print("Search terms: " + searchterms)
    print("Sending this query (Sorted by last-updated): " + query)
    col2expander.write("Sending this query (Sorted by last-updated): " + query)
    result = g.search_repositories(query, 'updated', 'desc')

    print(f'Found {result.totalCount} repo(s)')
    col2expander.write(f'Found {result.totalCount} repo(s)')
    col2expander.write('**********************************')

    for repotocheck in result:
        repourl = str(repotocheck.clone_url).replace(".git", "")
        reponame = str(repotocheck.name)
        repoowner = str(repotocheck.owner.login)
        repolastupdated = str(repotocheck.updated_at)
        print(repourl + f'{repotocheck.stargazers_count} stars')
        col2expander.write(repourl + f'{repotocheck.stargazers_count} stars')
        print("Checking repo: " + reponame)
        col2expander.write("Checking repo: " + reponame)
        print("Repo Owned by: " + repoowner)
        col2expander.write("Repo Owned by: " + repoowner)
        contributorsurl = str(repotocheck.contributors_url)
        githeaders = {
            'Authorization': 'token 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'}
        response = requests.request("GET", contributorsurl, headers=githeaders)
        #print (response.text)
        # try:
        #     contributorslist = json.loads(response.text)
        #     contributors, contributions = contributor_counts(contributorslist)
        # except:
        #     contributorslist = ''
        #     contributors = 0
        #     contributions = 0
        #contributors = contributor_count(contributorslist)

        # print ("There were a total of " + str(contributors) + " contributors!")
        # col2expander.write("There were a total of " + str(contributors) + " contributors!")
        # print ("There was a total of " + str(contributions) + " contributions!")
        # col2expander.write("There was a total of " + str(contributions) + " contributions!")
        #query = f'in:file extension:md'
        #query = '+user:' + reponame + '+in:file+extension:md'
        #query = f'"{keyword} english" in:file extension:po'
        #result = g.search_code(query, order='desc')
        # for file in result:
        # print(f'{file.download_url}')
        #repo = git.repo(repourl)
        #readmexists = fileInRepo(repo,'README.MD')
        #print ("README IS HERE: " + str(readmexists))
        # if (contributors >= repocontributors):
        githubtogitanswers('README.md', reponame, repourl, repoowner, repostars, repoforks, repopushdate, repolastupdated, searchterms)
        githubtogitanswers('*.pdf', reponame, repourl, repoowner, repostars, repoforks, repopushdate, repolastupdated, searchterms)
    col2.success("GitHub Research Complete! Loading documents for Cognitive Analysis into gitanswers NOW!")
    gitanswersvolumetoadd = '{ "inputs": [ { "connectorId": "'+S3connector + \
        '", "path": "'+S3ingestpath+'", "isDirectory": '+isS3folder+'}]}'
    gitanswersheaders = {
        'Accept': 'application/json',
        'Authorization': 'Bearer {}'.format(gitanswersapikey),
        'Content-Type': 'application/json',
    }
    volumesListing = {}
    # volumesURL = "https://" + gitanswersinstance + "/amber/v1/volumes"
    # print('Calling URL: ' + str(volumesURL))
    # col2expander.write('Calling URL: ' + str(volumesURL))
    # #getvolumeIDs = requests.get(volumesURL, headers=gitanswersheaders)
    # #print(gitanswersheaders)
    # print(getvolumeIDs.text)
    #volumeIdslist = json.loads(getvolumeIDs.text)
    #for volumeid in volumeIdslist['volumes']:
        #if 'volumeName' not in volumeid:
            #continue
        #print ('*********************************\n')
        #volumeDescript = str(volumeid['volumeName'])
        #VolumeID = str(volumeid['volumeId'])
        ##VolumeStatus = str(volumeid['status'])
        #print ('Volume Name: ' + str(volumeDescript) + '\n')
        #print ('volumeID: ' + str(VolumeID) + '\n')
        # print ('Volume Status: ' + str(#VolumeStatus) + '\n')
        #print ('*********************************\n')
        #volumesListing.update({volumeDescript: VolumeID})
    # chosenVolume = gitanswersVolumeName
    # chosenVolumestr = str(chosenVolume)
    # print('Looking up Volume-SET: ' + chosenVolumestr)
    # print('********************************************\n')
    # volumeId = str(volumesListing.get(chosenVolumestr))
    # print('You Chose VolumeID: ' + volumeId)
    # print('********************************************\n')
    # print('Sending gitanswers this data: ' + str(gitanswersvolumetoadd))
    # print('********************************************\n')
    # #volumeId = '79d8ea26-c232-4d4b-a825-adc2ce65fd74'
    # VolumeJobsURL = "https://" + gitanswersinstance + \
    #     "/amber/v1/volumes/" + volumeId + "/jobs"
    #response = requests.post(VolumeJobsURL, data=gitanswersvolumetoadd, headers=gitanswersheaders)
    #response = json.loads(response.text)
    #print(response)
    #parts = response_parsed["engine"]["cogKg"]["jobDesc"]["volumeId"]["status"]
    #parts = response_parsed[0]
    print("Here's What's Going On:\n")
    print("------------------------\n")
    #print('Engine used: ' + str(response['engine']))
    #print ("Job Status: " + str(response['status']))
    #KGStatus = str(response['status'])
    #print("Job ID: " + str(response['jobId']))
    #volumeId = str(response['volumeId'])
    #print("Job Description: GitHub READMEs added to Volume: " + gitanswersVolumeName)
    #jobStatus = str(response['status'])
    #print("Job Status: " + jobStatus)
    #jobStartTime = str(response['startedAt'])
    #print ("KG ID: " + str(response['cogspaceId']['cogVersionId']['cogKg']))
    print("------------------------\n")

#keywords = input('Enter keyword(s)[e.g python, flask, postgres]: ')



col1, col2 = st.columns(2)
st.sidebar.image(image, caption=caption, width=250, use_column_width=False,
                 clamp=False, channels='RGB', output_format='auto')
st.sidebar.title("GitAnswers\n GitHub Researcher")
username = st.sidebar.text_input('username')
password = st.sidebar.text_input('password', type='password')

#state.logintogitlit = st.sidebar.button("LOGIN")
# DB Management
conn = sqlite3.connect('streamuser.db', check_same_thread=False)
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
    # html('<p> HEY!!!!!!!!!!!!!!!!! </p>')
    # col1.button('Open GitAnswers QA Portal', on_click=open_page, args=('http://10.10.10.91:8502', ))    
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

    col2.title("GitAnswers\n GitHub Researcher")
    col2.subheader("Let's GIT some data.....")
    #st.write("gitanswers GitHub Researcher")
    image1 = './githubleaks.jpeg'
    image2 = './webspider.png'
    col2.image(image1, caption=None, width=None, use_column_width=True,
               clamp=False, channels='RGB', output_format='auto')
    # st.image(image2, caption=caption, width=None, use_column_width=True,
    # clamp=False, channels='RGB', output_format='auto')
    #st.sidebar.image(sidebarimage, caption=caption, width=50, use_column_width=False,clamp=False, channels='RGB', output_format='auto')
    gitanswersinfoexpander = col1.expander("gitanswers - Customer S3 Account Information:")
    with gitanswersinfoexpander:
        gitanswersinfoexpander.header("gitanswers Account Information:")
        gitanswersinstance = gitanswersinfoexpander.text_input(
            "gitanswers Instance (include port number if not on 443):", value="myinstance.gitanswers.com:3000 or myinstance.gitanswers.com")
        # hardcoding gitanswers instance for simplicity can be uncommented above for dynamic form-access
        #gitanswersinstance = "intel-wl-poc.gitanswers.com"
        S3bucketname = gitanswersinfoexpander.text_input(
            "Top Level S3 Bucket Name for Research Document Storage:", value='gitanswers-field')
        S3ingestpath = gitanswersinfoexpander.text_input(
            "S3 Folder to place Research Documents in: (gitanswers will auto-create this on S3 temporarily for initial ingestion - please use _ instead of spaces)", value="github_readmes_raw_plus_spider")
        S3connector = gitanswersinfoexpander.text_input(
            "gitanswers S3 Connector Name For gitanswers to use to Load the Research Documents:", value="filed")
        # hardcoding S3connector for simplicity can be uncommented above for dynamic form-access
        #S3connector = ""
        gitanswersVolumeName = gitanswersinfoexpander.text_input(
            "gitanswers Volume to Add Research Documents to:", value="Github_Readmes_PLUS")
        volumeName = gitanswersVolumeName
        gitanswersapikey = gitanswersinfoexpander.text_input(
            "gitanswers API Key for accessing gitanswers:", value="AAAQAAAAA3XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXMYHI2TTJBCXIMKUME======", type='password')
        # hardcoding API-Key for simplicity can be uncommented above for dynamic form-access
        
    col1.write("****************************************************")
    githubinfoexpander = col1.expander("GitHub Search Criteria:")
    with githubinfoexpander:
        githubinfoexpander.header("GitHub Search Criteria:")
        repopushdate = githubinfoexpander.date_input(
            'Earliest Repo Creation Date')
        repostars = githubinfoexpander.number_input(
            "# of Stargazers Repo Must Have:", value=1)
        repoforks = githubinfoexpander.number_input(
            "# of Forks Repo Must Have:", value=1)
        repomoddate = githubinfoexpander.date_input(
            'Last Modified Date')
        # repostars = sidebar.slider("# of Stargazers Repo Must Have",min_value=1,max_value=5000,value=1,step=1)
        repocontributors = githubinfoexpander.number_input(
            "# of Contributors Repo Must Have:", value=1)
        repoactive = githubinfoexpander.checkbox("Repo Has Been Archived")
        if (repoactive):
            repoactive = 'True'
        else:
            repoactive = 'False'

    @cache_on_sidebarbutton_press('CHANGE PASSWORD')
    def changepass(username, password, loggedin):
        if loggedin:
            print('hi')
    #top_k_retriever = st.sidebar.slider("Max. number of documents from retriever",min_value=1,max_value=10,value=3,step=1)
    repotopics = col2.text_input(
        "Please Enter 'keywords' to search for GitHub Repositories under that topic:", value="I.e. blockchain workload")
    run_query = col2.button("Run")
    #debug = st.sidebar.checkbox("Show debug info")
    if run_query:
        print("************************************")
        print("STARTING A NEW SPIDER SESSION!")
        print("************************************")
        with st.spinner("Gathering documents (README.md, docx, pdf, etc.) from Github Repositories and any references in the repositories to Load into gitanswers...\n "
                        "For Further Research Please Login into your gitanswers Sandbox at: \n"
                        "https://intel.gitanswers.com"):
            col2expander = col2.expander(
                "Research Progress Viewer (Press the '+' to Monitor Progress) ")
            with col2expander:
                keywords = [keyword.strip()
                            for keyword in repotopics.split(' ')]
                search_github(keywords, repopushdate, repomoddate, repostars, repoforks, repoactive, repocontributors, col2expander,
                              gitanswersinstance, s3bucketname, S3ingestpath, gitanswersapikey, S3connector, gitanswersVolumeName, volumeName, repolastupdated)
                
                col1.button('Open GitAnswers QA Portal', on_click=open_page, args=('http://10.10.10.91:8502', ))    
            #results,raw_json = retrieve_doc(repotopics,top_k_reader=top_k_reader,top_k_retriever=top_k_retriever)
        #st.write("**** Researching Repos on Github....Standby....")
        # for result in results:
            # annotate_answer(result['answer'],result['context'])
            #'**Relevance:** ', result['relevance'] , '**Source:** ' , result['source']
        # if debug:
            #col2.subheader('REST API JSON response')
            # col2expander.write(raw_json)
# else:
    #st.sidebar.warning("Incorrect Username/Password")
