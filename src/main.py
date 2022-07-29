from os.path import join
from time import time
from latent_lrr import latent_lrr
from matplotlib.pyplot import imshow, figure, axis, show
from numpy import maximum, minimum, asarray, uint8
from PIL import Image

index = int(input('Enter the picture number (1-16): '))  # Can be from 1-16

if not(0 < index < 17):
    print('Value needs to be 1-16')
    exit(-1)


path1 = join('./images/IR' + str(index) + '.png')
path2 = join('./images/VIS' + str(index) + '.png')
fuse_path = join('./images/fused/fused' + str(index) + '_latllr.png')

image1 = Image.open(path1)
image2 = Image.open(path2)

image1 = asarray(image1)
image2 = asarray(image2)

if len(image1.shape) == 3 and image1.shape[2] > 1:
    image1 = Image.fromarray(uint8(image1))
    image2 = Image.fromarray(uint8(image2))
    image1 = image1.convert('L')
    image2 = image2.convert('L')
    image1 = asarray(image1)
    image2 = asarray(image2)


image1 = image1.astype(float)
image1 = image1 / 255
image2 = image2.astype(float)
image2 = image2 / 255

lambda_value = 0.8

print('LatLLR: ')

tic = time()

X1 = image1
Z1, L1, E1 = latent_llr(X1, lambda_value)
X2 = image2
Z2, L2, E2 = latent_llr(X2, lambda_value)

toc = time()

print(f'Elapsed time = {toc - tic} seconds.')

print('LatLLR: ')

I_llr1 = X1.dot(Z1)
I_saliency1 = L1.dot(X1)
I_llr1 = maximum(I_llr1, 0)
I_llr1 = minimum(I_llr1, 1)
I_saliency1 = maximum(I_saliency1, 0)
I_saliency1 = minimum(I_saliency1, 1)
I_e1 = E1

I_lrr2 = X2.dot(Z2)
I_saliency2 = L2.dot(X2)
I_lrr2 = maximum(I_lrr2, 0)
I_lrr2 = minimum(I_lrr2, 1)
I_saliency2 = maximum(I_saliency2, 0)
I_saliency2 = minimum(I_saliency2, 1)
I_e2 = E2

F_llr = (I_llr1 + I_lrr2) / 2

F_saliency = I_saliency1 + I_saliency2

F = F_llr + F_saliency

axis('off')
figure(1)
imshow(I_saliency1)
figure(2)
imshow(I_saliency2)
figure(3)
imshow(F)

show()

result = Image.fromarray(uint8(F * 255))
result.save(fuse_path)
