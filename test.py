# importing csv module
import csv
# importing sqlite3 module
import sqlite3
import pandas as pd

# read the csv file
# with open('newSlide.csv' , 'r') as csvfile:
#     # create the object of csv.reader()
#     csv_file_reader = csv.reader(csvfile,delimiter=',')
#     # skip the header 
#     next(csv_file_reader,None)
#     # create fileds 
#     index =''
#     word=''
#     paragraphNo=''
#     titleNo=''
#     countRepeat = ''

    
#     ##### create a database table using sqlite3###
    
#     # 1. create query    
#     Table_Query = 'create table newSlide("index" int ,"word" varchar2(25), "paragraphNo" int, "titleNo" int , "countRepeat" int);'
    
# #     # 2. create database
connection=sqlite3.connect('db_csv.db')
curosr=connection.cursor()
#     # 3. execute table query to create table
#     curosr.execute(Table_Query)

#     # 4. pase csv data
#     for row in csv_file_reader:
#         # skip the first row
#         for i in range(len(row)):
#             # assign each field its value
#             index=row[0]
#             word=row[1]
#             paragraphNo=row[2]
#             titleNo=row[3]
#             countRepeat = row[4]
            
        
#         # 5. create insert query
#         InsertQuery=f"INSERT INTO newSlide VALUES ('{index}','{word}','{paragraphNo}','{titleNo}','{countRepeat}')"
#         # 6. Execute query
#         curosr.execute(InsertQuery)

# data=curosr.execute('''SELECT * FROM newSlide''')
data=curosr.execute('''SELECT * 
                            FROM newSlide co   
                                WHERE ( 
                                    SELECT COUNT(*) 
                                        FROM newSlide ci 
                                            WHERE  co.word = ci.word 
                                                    
                                                    AND co.countRepeat < ci.countRepeat ) 
                                                < 5''')

df = pd.DataFrame(data)
df.to_csv('test_max_select.csv', encoding='utf-8')

# for row in data:
#     print(row)
    # 7. commit changes
connection.commit()
    # 8. close connection
connection.close()

