import shutil
import os

destination = os.getcwd()
walker = os.walk(destination+'/fullset')

for data in walker:
	for files in data[2]:
		try:
			shutil.move(data[0]+'/'+files, destination)
			print files
		except shutil.Error:
			print files, 'fail'
			continue

print 'done'
