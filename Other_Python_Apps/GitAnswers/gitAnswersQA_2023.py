### AUTHOR: Neal Trieber
#####################################################
# * PRELIMINARY STEPS / PREP * ######################
#####################################################
# STEP 1) pip install -r requirements.txt --> included in repo!
# STEP 2) ENJOY!
##########***********************************************
# NOTES: *      - USES STREAMLIT as a UI  
#        *      - USES Deepset/Haystack For NLP Pipelining
##########***********************************************
from github import Github
import git
import re
import html2text
import requests
import sys
from annoy import AnnoyIndex
from concurrent.futures import ThreadPoolExecutor
import os, shutil
import os.path
import sqlite3
import csv
import sqlalchemy
import glob
import tabulate
import io
from io import StringIO as StringIO
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
from json import JSONDecodeError
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
import cohere
co = cohere.Client('GoXqyC9GZpFg3jU3gYRTTgf1nEImN65JGDAgPd0H') # This is your trial API key
# print('Summary:', cohere_response.summary)
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
import numpy as np
import string
from haystack.utils import print_documents, convert_files_to_docs
answer = ''
text = ''
context = ''
image = "githubleaks.jpeg"
caption = "RE-SEARCH. GIT Answers."
sidebarimage = "githubleaks.jpeg"
loggedin = ''
loginresult = ''
train_file = None
df = ''
prompt = ''
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

def process_text_input(text: str, run_id: str = None):  
	text = StringIO(text).read()  
	chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]  
	df = pd.DataFrame.from_dict({'text': chunks})  
	return df

def process_csv_file(train_file):  
	df = pd.read_csv(train_file, encoding='utf-8') 
	return df

def embed(list_of_texts):  
	response = co.embed(model='small', texts=list_of_texts)  
	return response.embeddings

def top_n_neighbors_indices(prompt_embedding: np.ndarray, storage_embeddings: np.ndarray, n: int = 5):  
	if isinstance(storage_embeddings, list):  
		storage_embeddings = np.array(storage_embeddings)  
	if isinstance(prompt_embedding, list):  
		storage_embeddings = np.array(prompt_embedding)  
	similarity_matrix = prompt_embedding @ storage_embeddings.T / np.outer(norm(prompt_embedding, axis=-1), norm(storage_embeddings, axis=-1))  
	num_neighbors = min(similarity_matrix.shape[1], n)  
	indices = np.argsort(similarity_matrix, axis=-1)[:, -num_neighbors:]  
	return indices

def chunk_text(df, width=1500, overlap=500):
    # create an empty dataframe to store the chunked text
    new_df = pd.DataFrame(columns=['id', 'text_chunk'])

    # iterate over each row in the original dataframe
    for index, row in df.iterrows():
        # split the text into chunks of size 'width', with overlap of 'overlap'
        chunks = []
        rows = []
        for i in range(0, len(row['Text']), width - overlap):
            chunk = row['Text'][i:i+width]
            chunks.append(chunk)

        # iterate over each chunk and add it to the new dataframe
        chunk_rows = []
        for i, chunk in enumerate(chunks):
            # calculate the start index based on the chunk index and overlap
            start_index = i * (width - overlap)

            # create a new row with the chunked text and the original row's ID
            new_row = {'id': row['id'], 'text_chunk': chunk, 'start_index': start_index}
            chunk_rows.append(new_row)
        chunk_df = pd.DataFrame(chunk_rows)
        new_df = pd.concat([new_df, chunk_df], ignore_index=True)
    return new_df
def search(query, n_results, df, search_index, co):
    # Get the query's embedding
    query_embed = co.embed(texts=[query],
                    model="f4977328-3655-413c-82ef-3520da58852f-ft",
                    truncate="LEFT").embeddings
    
    # Get the nearest neighbors and similarity score for the query and the embeddings, 
    # append it to the dataframe
    nearest_neighbors = search_index.get_nns_by_vector(
        query_embed[0], 
        n_results, 
        include_distances=True)
    # filter the dataframe to include the nearest neighbors using the index
    df = df[df.index.isin(nearest_neighbors[0])]
    index_similarity_df = pd.DataFrame({'similarity':nearest_neighbors[1]}, index=nearest_neighbors[0])
    df = df.join(index_similarity_df,) # Match similarities based on indexes
    df = df.sort_values(by='similarity', ascending=False)
    return df


# define a function to generate an answer
def gen_answer(q, para): 
    response = co.generate( 
        model='command-xlarge-20221108', 
        prompt=f'''Paragraph:{para}\n\n
                Answer the question using this paragraph.\n\n
                Question: {q}\nAnswer:''', 
        max_tokens=100, 
        temperature=0.4)
    return response.generations[0].text

def gen_better_answer(ques, ans): 
    response = co.generate( 
        model='command-xlarge-20221108', 
        prompt=f'''Answers:{ans}\n\n
                Question: {ques}\n\n
                Generate a new answer that uses the best answers 
                and makes reference to the question.''', 
        max_tokens=100, 
        temperature=0.4)
    return response.generations[0].text

def display(query, results):
    # 1. Run co.generate functions to generate answers

    # for each row in the dataframe, generate an answer concurrently
    with ThreadPoolExecutor(max_workers=1) as executor:
        results['answer'] = list(executor.map(gen_answer, 
                                              [query]*len(results), 
                                              results['text_chunk']))
    answers = results['answer'].tolist()
    # run the function to generate a better answer
    answ = gen_better_answer(query, answers)
    
    # 2. Code to display the resuls in a user-friendly format

    st.subheader(query)
    st.write(answ)
    # add a spacer
    st.write('')
    st.write('')
    st.subheader("Relevant documents")
    # display the results
    for i, row in results.iterrows():
        # display the 'Category' outlined
        st.markdown(f'**{row["Type"]}**')
        st.markdown(f'**{row["Category"]}**')
        st.markdown(f'{row["title"]}')
        # display the url as a hyperlink
        # add a button to open the url in a new tab
        st.markdown(f'[{row["link"]}]({row["link"]})')
        st.write(row['answer'])
        # collapse the text
        with st.expander('Read more'):
            st.write(row['text'])
        st.write('')




local_css("gitAnswers.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = ""
image = "githubleaks.jpeg"
caption = "RE-SEARCH. RE-DISCOVERED."
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
    st.write('*******************************')
    st.header('COHEREnt Summarization')     
    st.write('*******************************')
    question = st.text_input("Summarize THIS (Enter Text Here):", value=st.session_state.question, max_chars=100000, on_change=reset_results)
    summary_pressed = st.button("Summarize IT!")
    st.write('*******************************')
    st.header('Upload or Copy/Paste Files here for Search or Summarization:')
    st.write('*******************************')
    option = st.selectbox("Document Input:", ["TEXT BOX", "CSV", "READMES FRON GitAnswers Downloads"])  
    if option == "CSV":  
        train_file = st.file_uploader("Upload A Custom Training CSV File TO Make it more COHEREnt", help="Accepts a two column csv", type=["csv"])  
        embeddings = None  
    if train_file is not None:  
            df, _, _, _ = process_csv_file(train_file)  
    if option == "TEXT BOX":  
        text = st.text_area("Paste a Document HERE")  
        if text != "":  
            df = process_text_input(text)
    if option == "READMES FRON GitAnswers Downloads":      
        doc_dir = "./newreadmes/"
        txtfiles = []		
#print(glob.glob("/Users/neal.trieber/Downloads/Promed2012_2014"))
        with open('data_to_import.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            Colheader = ['Text']
            writer.writerow(Colheader)
            out_file.close()
              
        for file in glob.glob(doc_dir + "*.txt"):
            txtfiles.append(file)
            print (file)
            with open(file, 'r') as in_file:
                stripped = (line.strip() for line in in_file)
                result = " ".join(line.strip() for line in in_file)
                print (result)
                lines = (line.split(",") for line in stripped if line)
                # print (lines[1] + ', ' + lines[2])
                with open('data_to_import.csv', 'a') as myout_file:
                    writer = csv.writer(myout_file)
                    if result is not "":
                        writer.writerow([result])
                    print ('FILE INGESTED!')
                    myout_file.close()
        # dicts = convert_files_to_docs(dir_path=doc_dir)
        # print(dicts[:3])
        df = pd.read_csv('data_to_import.csv')
        print (df)
        # add an id column
        df['id'] = df.index
        new_df = chunk_text(df)
        # append text chunks to the original dataframe in id order
        df = df.merge(new_df, on='id', how='left')
        # df
        # Get the embeddings
        embeds = co.embed(texts=list(df['text_chunk']),
                        model="large",
                        truncate="RIGHT").embeddings
        # Check the dimensions of the embeddings
        embeds = np.array(embeds)
        embeds.shape
        # Create the search index, pass the size of embedding
        search_index = AnnoyIndex(embeds.shape[1], 'angular')
        # Add all the vectors to the search index
        for i in range(len(embeds)):
            search_index.add_item(i, embeds[i])

        search_index.build(10) # 10 trees
        search_index.save('search_index.ann')
        # export the dataframe to a csv file
        df.to_csv('cohere_text_import.csv', index=False)
        
        # Load the search index
        search_index = AnnoyIndex(f=4096, metric='angular')
        search_index.load('search_index.ann')

        # load the csv file called cohere_final.csv
        df = pd.read_csv('cohere_text_import.csv')



    if question:
        cohere_response = co.summarize( 
            text=question,
            length='auto',
            format='auto',
            model='summarize-xlarge',
            additional_command='',
            temperature=0.9,
            )
        
    # if df is not NotImplemented:
    #     cohere_response = co.summarize( 
    #         text=question,
    #         length='auto',
    #         format='auto',
    #         model='summarize-xlarge',
    #         additional_command='',
    #         temperature=0.9,
    #         )    


    
    run_summary = (
        summary_pressed or question != st.session_state.question
    )


    # # Search bar
    
    if run_summary and question:
        reset_results()
        st.session_state.question = question
        with st.spinner("🔎 &nbsp;&nbsp; Reading Your Document...Summarization in Progress"):
            try:
                st.session_state.results = cohere_response.summary            
            except JSONDecodeError as je:
                st.error(
                    "👓 &nbsp;&nbsp; JSON DECODER had An error occurred reading the results. Is the document store working?"
                )    
            except Exception as e:
                logging.exception(e)
                st.error("🐞 &nbsp;&nbsp; A general error occurred during the request.")
            
                

    if st.session_state.results:
        st.write('## Here\'s what I read about:')
        st.write('*******************************')
        summary = st.session_state.results
        # count = range(len(answers))
        print ("HAZZAH IT WORKED! " + str(summary))
            # for x, y in ananswer.items():
            #     #print(key, '->', value)
            
            #     # print (answerparts[j])
            #     if (x == "answer"):
            #         text = y
            #     if (x == "context"):
            #         context = y
                    
                # start_idx = context.find(text)
                # end_idx = start_idx + len(text)
                # print ("***********************************")
                # print (str(context.find(text)))
                # print (context[:start_idx])
                # print (context[end_idx:])
                # print (text)
                # print ("***********************************")
                # st.write ("*****************************")
        st.markdown(str(annotation(body=text, label="SUMMARY", background="#964448", color='#ffffff')) + summary, unsafe_allow_html=True)
        st.write ("*****************************")           
        # st.write(
        #     markdown.markdown(str(annotation(body=text, label="SUMMARY", background="#964448", color='#ffffff')) + summary),
        #     unsafe_allow_html=True,
        # )
                #     # else:
                #     #     st.info("🤔 &nbsp;&nbsp; Haystack is unsure whether any of the documents contain an answer to your question. Try to reformulate it!")


    if run_summary and df is not None:
        reset_results()
        st.session_state.question = question
        with st.spinner("🔎 &nbsp;&nbsp; Reading Your Document...Summarization in Progress"):
            try:
                st.session_state.results = cohere_response.summary            
            except JSONDecodeError as je:
                st.error(
                    "👓 &nbsp;&nbsp; JSON DECODER had An error occurred reading the results. Is the document store working?"
                )    
            except Exception as e:
                logging.exception(e)
                st.error("🐞 &nbsp;&nbsp; A general error occurred during the request.")
            
                

    if st.session_state.results:
        st.write('## Here\'s what I read about:')
        st.write('*******************************')
        summary = st.session_state.results
        # count = range(len(answers))
        print ("HAZZAH IT WORKED! " + str(summary))
            # for x, y in ananswer.items():
            #     #print(key, '->', value)
            
            #     # print (answerparts[j])
            #     if (x == "answer"):
            #         text = y
            #     if (x == "context"):
            #         context = y
                    
                # start_idx = context.find(text)
                # end_idx = start_idx + len(text)
                # print ("***********************************")
                # print (str(context.find(text)))
                # print (context[:start_idx])
                # print (context[end_idx:])
                # print (text)
                # print ("***********************************")
                # st.write ("*****************************")
        st.markdown(str(annotation(body=text, label="SUMMARY", background="#964448", color='#ffffff')) + summary, unsafe_allow_html=True)
        st.write ("*****************************")           
        # st.write(
        #     markdown.markdown(str(annotation(body=text, label="SUMMARY", background="#964448", color='#ffffff')) + summary),
        #     unsafe_allow_html=True,
        # )
                #     # else:
                #     #     st.info("🤔 &nbsp;&nbsp; Haystack is unsure whether any of the documents contain an answer to your question. Try to reformulate it!")


    
    
    # Get results for query

    st.write ("*****************************")              
    st.header('COHEREnt Prompt - \'GitAnswersGPT\'')
    st.write ("*****************************")

    if 'question3' not in st.session_state:
        st.session_state['question3'] = ''
    if 'results3' not in st.session_state:
        st.session_state['results3'] = ''
            
    question3 = st.text_input("Ask Me Anything:", value=st.session_state.question3, max_chars=100000, on_change=reset_results)
    print (question3)
    if question3:
        cohere_promptresponse = co.generate(
        model='command',
        prompt=question3,
        max_tokens=3000,
        temperature=0.9,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE')
        
    prompt_pressed = st.button("Ask me Anything!")
    
    run_prompt = (
        prompt_pressed or question3 != st.session_state.question3
    )


    # # Search bar
    
    if prompt_pressed and question3:
        reset_results()
        st.session_state.question3 = question3
        with st.spinner("🔎 &nbsp;&nbsp; Let me thing about thought for a sec..."):
            try:
                st.session_state.results3 = cohere_promptresponse.generations[0].text            
            except JSONDecodeError as je:
                st.error(
                    "👓 &nbsp;&nbsp; JSON DECODER had An error occurred reading the results. Is the document store working?"
                )    
            except Exception as e:
                logging.exception(e)
                st.error("🐞 &nbsp;&nbsp; A general error occurred during the request.")
            
                

    if st.session_state.results3:
        reset_results()
        print('Prediction: {}'.format(cohere_promptresponse.generations[0].text))              
        prompt = 'Prediction: {}'.format(cohere_promptresponse.generations[0].text)
        st.write('## Here\'s What I think:')
        st.write('*******************************')
        st.markdown(str(annotation(body=text, label="Possibilities", background="#964448", color='#ffffff')) + prompt, unsafe_allow_html=True)
        st.write ("*****************************")           
        # st.write(
        #     markdown.markdown(str(annotation(body=text, label="Possibilites", background="#964448", color='#ffffff')) + prompt),
        #     unsafe_allow_html=True,
        #     )
    
    if 'question2' not in st.session_state:
        st.session_state['question2'] = ''
    if 'results2' not in st.session_state:
        st.session_state['results2'] = ''
    question2 = st.text_input("Ask A Question About your Github Readmes", value=st.session_state.question2, max_chars=1000, on_change=reset_results)
    # if 'question2' not in st.session_state:
    #     st.session_state['question2'] = 'question2'
            
    run_pressed = st.button("Submit Question")

    run_query = (
        run_pressed or (question2 != st.session_state.question2)
    )


    if question2:
        results = search(question2, 3, df, search_index, co)
        display(question2, results)

    if run_query and question2:
        reset_results()
        st.session_state.question2 = question2
        with st.spinner("🔎 &nbsp;&nbsp; Running your pipeline"):
            try:
                st.session_state.results2 = askquestion(question2)        
            except JSONDecodeError as je:
                st.error(
                    "👓 &nbsp;&nbsp; JSON DECODER had An error occurred reading the results. Is the document store working?"
                )    
            except Exception as e:
                logging.exception(e)
                st.error("🐞 &nbsp;&nbsp; A general error occurred during the request.")
            
    

    if st.session_state.results2:
        st.write('## We have The Answers for You:')
        st.write('*******************************')
        answers = st.session_state.results2
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


    if df is not None and prompt != "":
        base_prompt = "Based on the Documents Provided, answer the following question:"
        prompt_embedding = embed_stuff([prompt])
        aug_prompts = get_augmented_prompts(np.array(prompt_embedding), embeddings, df)
        new_prompt = '\n'.join(aug_prompts) + '\n\n' + base_prompt + '\n' + prompt + '\n'
        print(new_prompt)
        is_success = False
        while not is_success:
            try:
                response = generate(new_prompt)
                is_success = True
            except Exception:
                aug_prompts = aug_prompts[:-1]
                new_prompt = '\n'.join(aug_prompts) + '\n' + base_prompt + '\n' + prompt  + '\n'

        st.write(response.generations[0].text)