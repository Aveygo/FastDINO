import json

with open('similarity.json') as f:
    similarity = json.load(f)

with open('names.json') as f:
    name_key = json.load(f)

query = "/home/greg/Documents/Music/Music/corn wave.UCszXggkRMQOAly3y9_NVqbQ/dudeness/domestic wolves.mQQuGxePBvc.mp3"

query_idx = name_key.index(query)

similar = similarity[str(query_idx)]

print(query)
print("")

for i in similar:
    print(name_key[i])