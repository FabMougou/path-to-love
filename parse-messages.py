from imessage_tools import read_messages
import csv
import secret

db = secret.db
number = secret.number
messages = read_messages(db, None, 'Me', number, True)
headers = list(messages[0].keys())

with open('messages.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    writer.writerows(messages)