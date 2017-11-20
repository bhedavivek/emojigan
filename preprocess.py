import json
import os

data = json.load(open('emoji.json', 'r'))

temp = []

for obj in data:
	temp_obj = {}
	temp_obj['location'] = "./images/jpg/" + obj['unified'].lower() + ".jpg"
	if os.path.isfile(temp_obj['location']):
		if obj['name'] == None:
			os.remove(temp_obj['location'])
		else:
			temp_obj['annotation'] = obj['name'].lower()
			temp.append(temp_obj)

with open('processed.json', 'w+') as file:
	json.dump(temp, file, indent=4)