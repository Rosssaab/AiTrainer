import pyodbc

print("Available SQL Server Drivers:")
for driver in pyodbc.drivers():
    print(driver) 