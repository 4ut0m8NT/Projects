# streamlit run ./userstream.py to create an authentication DB with users.
import streamlit as st
import pandas as pd


# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
import sqlite3 
conn = sqlite3.connect('streamuser.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data


def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data



def main():
	"""Simple Login App"""
	image = "githubleaks.jpg"
	caption = "RE-SEARCH. RE-DISCOVERED."
	sidebarimage = "githubleaks.jpg"

	st.image(image, caption=caption, width=None, use_column_width=True,
          clamp=False, channels='RGB', output_format='auto')
	st.title("GitAnswers GitHub Researcher")

	menu = ["Login","Research Options"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Please Login.")

	elif choice == "Login":
		st.subheader("Please Login.")

		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')
		
		logintogitlit = st.sidebar.button("LOGIN")
		if logintogitlit:
			# if password == '12345':
			create_usertable()
			hashed_pswd = make_hashes(password)

			result = login_user(username,check_hashes(password,hashed_pswd))
			if result:
				#st.empty
				st.success("Logged In as {}".format(username))
				st.subheader("Let's Start Researching.")

				
			else:
				st.warning("Incorrect Username/Password")





	elif choice == "Research Options":
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")



if __name__ == '__main__':
    	main()
