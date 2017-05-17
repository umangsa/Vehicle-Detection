import glob

'''
	Read the training data from the supplied set
	return cars and non car data
'''
def read_training_data():
	cars = []
	noncars = []
	for filename in glob.iglob('training_data/vehicles/**/*.png', recursive=True):
		cars.append(filename)

	for filename in glob.iglob('training_data/non-vehicles/**/*.png', recursive=True):
		noncars.append(filename)

	return cars, noncars
