import streamlit as st
import streamlit_authenticator as stauth

# Pre-hash the valid password (this is a one-time operation)
valid_password = "hello"
hashed_password = stauth.Hasher([valid_password]).generate()[0]