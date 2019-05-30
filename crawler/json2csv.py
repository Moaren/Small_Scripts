import csv
import json
import pandas as pd

json_file = "1-240-273-1357.json"
with open(json_file, "r") as x:
    x = json.load(x)
print(len(x))

with open("test.csv", 'w', newline='', encoding='utf-8') as csv_file:
    f = csv.writer(csv_file)
    # Write CSV Header, If you dont need that, remove this line
    f.writerow(["content","phone_number","datetime", "caller", "call_type"])
    for comment in x:
        f.writerow([comment["content"],
                    json_file.split(".json")[0],
                    comment["datetime"],
                    comment["caller"],
                    comment["call_type"]])
df = pd.read_csv("test.csv")
for i in range(1,40):
    print(df.iloc[i])


