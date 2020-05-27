"""Test code for sqlite3 database maniplations"""

import sqlite3

conn = sqlite3.connect('example.db')


c = conn.cursor()

# Print the columns in our table
table = 'test'
c.execute('PRAGMA table_info({0})'.format(table))
results = c.fetchall()
for row in results:
    print(row[1])

# Create table
c.execute('''CREATE TABLE test
             (label1 text, label2 text, value1 real, value2 real)''')

# Insert a row of data
c.execute("INSERT INTO test VALUES ('bah1','bah2',567,5.6)")

# Save (commit) the changes
conn.commit()

# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()
