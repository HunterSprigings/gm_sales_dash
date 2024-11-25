import bcrypt

# Define plain text passwords
passwords = ["hello"]

# Convert plain text passwords to hashed passwords
hashed_password = bcrypt.hashpw(passwords[0].encode(), bcrypt.gensalt()).decode()
