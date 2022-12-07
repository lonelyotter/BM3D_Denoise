imgG = imread("../img/noise/gaussian.png");
imgP = imread("../img/noise/poisson.png");
imgSP = imread("../img/noise/salt_pepper.png");

imwrite(nlmFilter(imgG), "../img/non_local_means/gaussian.png")
imwrite(nlmFilter(imgP), "../img/non_local_means/poisson.png")
imwrite(nlmFilter(imgSP), "../img/non_local_means/salt_pepper.png")

function [output] = nlmFilter(img)

R=img(:,:,1);
G=img(:,:,2);
B=img(:,:,3);
R = myNlmFilter(R);
G = myNlmFilter(G);
B = myNlmFilter(B);
output = uint8(cat(3, R, G, B));

end

function [res] = myNlmFilter(img)
img = double(img);
t = 10;
f = 3;
sigma = 0.012;
[m, n] = size(img);
img2 = padarray(img, [f f], "symmetric");
res = zeros(m, n);

for i = 1 : m
    for j = 1 : n

        % 当前目标像素的坐标
        i1 = i + f;
        j1 = j + f;

        % 当前目标像素所在的图像块
        w1 = img2(i1 - f : i1 + f, j1 - f : j1 + f);

        % 窗口中其他图像块中心像素坐标的范围
        rmin = max(i1 - t,f + 1);
        rmax = min(i1 + t,m + f);
        smin = max(j1 - t,f + 1);
        smax = min(j1 + t,n + f);

        zx = 0;
        tot = 0;

        for r = rmin : rmax
            for s = smin : smax

                % 跳过目标像素
                if (r == i1 && s == j1)
                    continue;
                end

                w2 = img2(r - f : r + f, s - f : s + f);
                degree = exp((-sum(sum((w1 - w2) .^ 2))) / 2 * (sigma ^ 2));
                tot = tot + degree * img2(r, s);
                zx = zx + degree;
                
            end
        end
        res(i, j) = uint8(tot / zx);
    end
end

end