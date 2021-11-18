from Fetch_info import Parameters_collection
from sketch import Fisher_analysis

configfile = "/path/to/the/config.yaml"

pipeline_info = Parameters_collection.from_config(configfile) # Fetch info about the telescope and former steps

testclass = Fisher_analysis(0,0.3,2,0,0.15,2,pipeline_info)

Fisher = testclass.make_fisher(self, svd_cut="True")
