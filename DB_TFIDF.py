
import pandas as pd
from pathlib import Path
import sqlite3
import time

start_time = time.time()
# Execute a query that’ll create a users table with user_id and username columns

print("Reading Data..")
data = pd.read_csv("outputs/output_dev42.csv")
# data = pd.read_hdf('outputs/output_sampleH.h5', 'df')

# Create a database connection and cursor to execute queries
conn = sqlite3.connect('data_db.db')
c = conn.cursor()
c.execute('''DROP TABLE data_set''')
c.execute('''CREATE TABLE data_set (question text,
                                    word text,
                                    TF_IDF float,
                                    paragraph_no int,
                                    title_no int
                                    )''')

end_time = time.time()
print("execution time -> ", (end_time-start_time)/60, " minutes")

###############################################################################

start_time = time.time()
data.to_sql('data_set', conn, if_exists='append', index=False)

# for row in c.fetchall():
#     # can convert to dict if you want:
#     print(dict(row))
# data = c.execute('''SELECT * FROM data_set''')

print("Selecting data from database..")

max_select = c.execute('''SELECT question, word, MAX(TF_IDF), paragraph_no, title_no
                            FROM data_set
                                GROUP BY question, word''')
# ORDER BY question''')

end_time = time.time()
print("execution time -> ", (end_time-start_time)/60, " minutes")

###############################################################################

start_time = time.time()

print("Creating data frame..")
df = pd.DataFrame(max_select)
end_time = time.time()
print("execution time -> ", (end_time-start_time)/60, " minutes")

###############################################################################

start_time = time.time()
print('Creating .csv file..')
df.to_csv('MaxSelects/max_dev42.csv', encoding='utf-8', index=False)
# df.to_hdf('MaxSelects/data_sample.csv', encoding='utf-8',
#           key='df', index=False, mode='w')

end_time = time.time()
print("execution time -> ", (end_time-start_time)/60, " minutes")
print('End of "DB_main.py"')

conn.commit()
conn.close()
