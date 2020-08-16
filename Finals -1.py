
from shutil import copyfile
# define location of dataset
folder = 'Train/Red/'
# plot first few images
for i in range(300):
	for file in folder:
            src = folder + file
            dst = folder + 'Red.'+str(i)+'.jpg'
            copyfile(src, dst)
        