import sqlite3
from employee import Employee

conn = sqlite3.connect('employee.db')

c = conn.cursor()

# c.execute("""CREATE TABLE employees (
#                 first text,
#                 last text,
#                 pay integer
#             )""")

emp_1 = Employee('John', 'Doe', 70000)
emp_2 = Employee('Jim', 'Wang', 30000)

# Correct way 1 to avoid sql attack
c.execute("INSERT INTO employees VALUES (?, ?, ? )", (emp_1.first,emp_1.last,emp_1.pay))
conn.commit()

# Way 2
c.execute("INSERT INTO employees VALUES (:first, :last, :pay )",
          {"first":emp_2.first,"last":emp_2.last,"pay":emp_2.pay})
conn.commit()

c.execute("SELECT * FROM employees WHERE last = ?",("Park",))
print(c.fetchall())

c.execute("SELECT * FROM employees WHERE last = :last",{"last": "Doe"})
print(c.fetchall())

conn.close()