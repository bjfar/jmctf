"""Small helper functions for interacting with sqlite3 databases"""

import numpy as np

def create_table(cursor,table_name,columns):
    """Create an SQLite table if it doesn't already exist"""
    command = "CREATE TABLE IF NOT EXISTS `{0}` (`{1}` {2}".format(table_name,columns[0][0],columns[0][1])
    for col, t in columns[1:]:
        command += ",`{0}` {1}".format(col,t)
    command += ")"  
    #print("command:",command)
    cursor.execute(command)

def add_columns(cursor,table_name,columns):
    """Add columns to an SQLite table if they don't already exist"""
    # Check what columns currently exist
    cursor.execute('PRAGMA table_info({0})'.format(table_name))
    results = cursor.fetchall()
    existing_cols = [row[1] for row in results]
    # Have to add columns one at a time. Oh well, apparently should be fast anyway.
    commands = ["ALTER TABLE `{0}` ADD COLUMN `{1}` {2}".format(table_name,col,t) for col,t in columns if col not in existing_cols]
    for command in commands:
       cursor.execute(command)

def upsert(cursor,table_name,df,primary):
    """Insert or overwrite data into a set of SQL columns.
       Data assumed to be a pandas dataframe. Must specify
       which column contains the primary key.
    """
    rec = df.to_records()
    #print("rec:",rec)
    columns = df.to_records().dtype.names  
    command = "INSERT INTO `{0}` (`{1}`".format(table_name,columns[0])
    if len(columns)>1:
        for col in columns[1:]:
            command += ",`{0}`".format(col)
    command += ") VALUES (?"
    if len(columns)>1:
        for col in columns[1:]:
            command += ",?"
    command += ")"
    if len(columns)>1:
        command += " ON CONFLICT(`{0}`) DO UPDATE SET ".format(primary)
        for j,col in enumerate(columns):
            if col is not primary:
                command += "`{0}`=excluded.`{0}`".format(col)
                if j<len(columns)-1: command += ","
    #print("command:", command)
    cursor.executemany(command,  map(tuple, rec.tolist())) # sqlite3 doesn't understand numpy types, so need to convert to standard list. Seems fast enough though.

def upsert_if_smaller(cursor,table_name,df,primary):
    """As sql_upsert, but only replaces existing data if new values are smaller than those already recorded.
    """
    rec = df.to_records()
    #print("rec:",rec)
    columns = df.to_records().dtype.names  
    command = "INSERT INTO `{0}` (`{1}`".format(table_name,columns[0])
    if len(columns)>1:
        for col in columns[1:]:
            command += ",`{0}`".format(col)
    command += ") VALUES (?"
    if len(columns)>1:
        for col in columns[1:]:
            command += ",?"
    command += ")"
    if len(columns)>1:
        command += " ON CONFLICT(`{0}`) DO UPDATE SET ".format(primary)
        for j,col in enumerate(columns):
            if col is not primary:
                command += "`{0}` = CASE".format(col)
                command += "      WHEN `{0}`<excluded.`{0}` THEN `{0}`".format(col)
                command += "      ELSE excluded.`{0}`".format(col)
                command += "      END"
                if j<len(columns)-1: command += ","
    #print("command:", command)
    cursor.executemany(command,  map(tuple, rec.tolist())) # sqlite3 doesn't understand numpy types, so need to convert to standard list. Seems fast enough though.

def insert_as_arrays(cursor,table_name,df):
    """Do not treat Pandas rows as SQL rows; just dump each 
       column into one SQL entry as BLOB data"""

    columns = df.to_records().dtype.names[1:] # remove index column, don't care about it for this  
    command = "INSERT INTO {0} ({1}".format(table_name,columns[0])
    if len(columns)>1:
        for col in columns[1:]:
            command += ",{0}".format(col)
    command += ") VALUES (?"
    if len(columns)>1:
        for col in columns[1:]:
            command += ",?"
    command += ")"

    cursor.execute(command, df.to_numpy().T) # Storing entire dataframe column as one entry in an SQL row


def load(cursor,table_name,cols,keys=None,primary=None):
    """Load data from an sql table
       with simple selection of items by primary key values.
       Compressed primary key values into a set of ranges to
       construct more efficient queries.
       Assumes the primary key values are supplied 
       in ascending order. If keys is None, all rows are loaded."""

    if keys is not None:
        splitdata = np.split(keys, np.where(np.diff(keys) != 1)[0]+1)
        ranges = [(np.min(x), np.max(x)) for x in splitdata]

    if len(cols)==0:
        msg = "SQL load from table `{0}` failed: no columns specified! (cols={1})".format(table_name,cols)
        raise ValueError(msg)

    # Check columns
    #c.execute('PRAGMA table_info({0})'.format(table_name))
    #results = c.fetchall()
    print("cols: ", cols)

    command = "SELECT "
    for col in cols:
        command += "`{0}`,".format(col)
    command = command[:-1]
    command += " from `{0}`".format(table_name)

    if keys is not None:
        command += " WHERE "
        for i,(start, stop) in enumerate(ranges):
            command += " `{0}` BETWEEN {1} and {2}".format(primary,start,stop) # inclusive 'between' 
            if i<len(ranges)-1: command += " OR "
    print("command:", command)
    cursor.execute(command)
    return cursor.fetchall() 

def table_info(cursor,table):
    """Return table metadata"""
    command = "PRAGMA table_info(`{0}`)".format(table)
    cursor.execute(command)
    return cursor.fetchall()

def check_table_exists(cursor,table):
    """Check if a table exists in the database"""
    command = "SELECT name FROM sqlite_master WHERE type = 'table'"
    cursor.execute(command)
    results = cursor.fetchall()
    if table in [r[0] for r in results]: 
        return True
    else:
        return False

