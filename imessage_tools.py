import sqlite3
import datetime
import subprocess
import os

def read_messages(db_location, n=10, self_number='Me', other_number=None, human_readable_date=True):
    conn = sqlite3.connect(db_location)
    cursor = conn.cursor()

    query = """
    SELECT message.ROWID, message.date, message.date_delivered, message.date_read, message.text, message.attributedBody, handle.id, message.is_from_me
    FROM message
    LEFT JOIN handle ON message.handle_id = handle.ROWID
    """

    if other_number is not None:
        query += f" WHERE handle.id = '{other_number}'"
        
    if n is not None:
        query += f" ORDER BY message.date DESC LIMIT {n}"
    
        

    results = cursor.execute(query).fetchall()
    messages = []

    for result in results:
        rowid, date, date_delivered, date_read, text, attributed_body, handle_id, is_from_me = result

        if handle_id is None:
            phone_number = self_number
        else:
            phone_number = handle_id

        if text is not None:
            body = text
        elif attributed_body is None:
            continue
        else:
            attributed_body = attributed_body.decode('utf-8', errors='replace')

            if "NSNumber" in str(attributed_body):
                attributed_body = str(attributed_body).split("NSNumber")[0]
                if "NSString" in attributed_body:
                    attributed_body = str(attributed_body).split("NSString")[1]
                    if "NSDictionary" in attributed_body:
                        attributed_body = str(attributed_body).split("NSDictionary")[0]
                        attributed_body = attributed_body[6:-12]
                        body = attributed_body

        if human_readable_date:
            date = get_human_readable_date(date)
            date_delivered = get_human_readable_date(date_delivered)
            date_read = get_human_readable_date(date_read)
            
        messages.append(
            {"rowid": rowid, "date": date, "date_delivered" : date_delivered, "date_read" : date_read, "body": body, "phone_number": phone_number, "is_from_me": is_from_me})

    conn.close()
    return messages


def print_messages(messages):
    for message in messages:
        print(f"RowID: {message['rowid']}")
        print(f"Body: {message['body']}")
        print(f"Phone Number: {message['phone_number']}")
        print(f"Is From Me: {message['is_from_me']}")
        print(f"Cache Roomname: {message['cache_roomname']}")
        print(f"Group Chat Name: {message['group_chat_name']}")
        print(f"Date: {message['date']}")
        print("\n")

def get_human_readable_date(date):
    date_string = '2001-01-01'
    mod_date = datetime.datetime.strptime(date_string, '%Y-%m-%d')
    unix_timestamp = int(mod_date.timestamp())*1000000000
    new_date = int((date+unix_timestamp)/1000000000)
    date = datetime.datetime.fromtimestamp(new_date).strftime("%Y-%m-%d %H:%M:%S")
    
    return date