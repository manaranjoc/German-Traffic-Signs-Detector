import click
import os
from sklearn import linear_model

@click.group()
def cli():
	pass

@cli.command('download')
def download():
	import urllib.request
	import zipfile

	urllib.request.urlretrieve ("http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip", "images/FullIJCNN2013.zip")
	zipref = zipfile.ZipFile("images/FullIJCNN2013.zip",'r')
	zipref.extractall("images")
	zipref.close()
	os.remove("images/FullIJCNN2013.zip")

@cli.command("order")
def order():
	import random
	import shutil
	my_files = []
	for path, subdirs, files in os.walk('images/FullIJCNN2013'):
		for name in files:
			my_files.append(os.path.join(path, name).replace("images/FullIJCNN2013\\",""))
	my_files = my_files[902:]
	choice = random.sample(my_files, 970)
	for x in range(0,10):
		os.mkdir("images/train/0%i"% (x))
		os.mkdir("images/test/0%i"% (x))
	for x in range(10,43):
		os.mkdir("images/train/%i"% (x))
		os.mkdir("images/test/%i"% (x))
	for x in choice:
		os.rename("images/FullIJCNN2013/"+x,"images/train/"+x)
		my_files.remove(x)
	for x in my_files:
		os.rename("images/FullIJCNN2013/"+x,"images/test/"+x)
	shutil.rmtree("images/FullIJCNN2013")
	os.remove("images/FullIJCNN2013.zip")

@cli.command("train")
@click.option('--model', '-m', default="models/model1/saved/model1.pkl", help="Model to chose")
@click.option('--directory', '-d', default="images/train", help="Folder with data to train the model.")
def train(model, directory):

	data, target, images = transform(directory)
	
	from sklearn.externals import joblib

	logistic = joblib.load(model)

	print('LogisticRegression score: %f'
      % logistic.fit(data, target).score(data, target))

	
	joblib.dump(logistic, model)

@cli.command("test")
@click.option('--model', '-m', default="models/model1/saved/model1.pkl", help="Model to chose")
@click.option('--directory', '-d', default="images/test", help="Folder with data to test the model.")
def test(model, directory):
	
	data, target, images = transform(directory)
	
	from sklearn.externals import joblib

	logistic = joblib.load(model)

	print('LogisticRegression for test score: %f'
      % logistic.score(data, target))

@cli.command("infer")
@click.option('--model', '-m', default="models/model1/saved/model1.pkl", help="Model to chose")
@click.option('--directory', '-d', default="images/user", help="Folder that holds images to infer with the model.")
def infer(model, directory):
	import imageio
	import numpy
	import math
	from skimage import color
	from skimage.transform import resize
	from sklearn.externals import joblib
	import matplotlib.pyplot as plt

	logistic = joblib.load(model)

	images = []
	for path, subdirs, files in os.walk(directory):
		for name in files:
			pathT = os.path.join(path, name)
			imTemp = imageio.imread(pathT)
			imTemp = color.rgb2gray(imTemp)
			imTemp = resize(imTemp, (20,20))
			images.append(imTemp)

	n_samples = len(images)
	data = numpy.reshape(images, (n_samples,-1))

	prediction = logistic.predict(data)
	cont = 1
	number = 1
	for index in range(0,n_samples):
		plt.subplot(math.ceil(n_samples*2/6),6,index+cont)
		plt.axis('off')
		plt.imshow(images[index], cmap='gray')
		plt.title('Prediction: %i' % prediction[index])
		
		if number%6==0:
			cont = cont
		elif number%3==0:
			cont = cont+2
		else:
			cont = cont+1
		number= number +1;

	plt.show()

#Function for transforming images
def transform(directory):
	import imageio
	import numpy
	from skimage import color
	from skimage.transform import resize
	images = []
	target = []
	for path, subdirs, files in os.walk(directory):
		for name in files:
			pathT = os.path.join(path, name)
			imTemp = imageio.imread(pathT)
			imTemp = color.rgb2gray(imTemp)
			imTemp = resize(imTemp, (20,20))	
			images.append(imTemp)
			target.append(int(os.path.basename(os.path.dirname(pathT))))
	
	n_samples = len(images)
	data = numpy.reshape(images, (n_samples,-1))
	return data, target, images
if __name__ == '__main__':
    cli()