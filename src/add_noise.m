image = imread("../img/original/lena.png");
imageG = imnoise(image, 'gaussian', 0, 0.005);
imageP = imnoise(image, 'poisson');
imageSP = imnoise(image, 'salt & pepper', 0.01);
imwrite(imageG, "../img/noise/gaussian.png");
imwrite(imageP, "../img/noise/poisson.png");
imwrite(imageSP, "../img/noise/salt_pepper.png");
