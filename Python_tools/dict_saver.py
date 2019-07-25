import json

class dict_saver:
	def __init__(self,dic):
		self.dic = dic
		pass

	def to_json(self, file_name= "converted.json"):
		with open(file_name,"w+") as f:
			json.dump(self.dic,f)

if __name__ == '__main__':
	dic = {1:2}
	saver = dict_saver(dic)
	saver.to_json()