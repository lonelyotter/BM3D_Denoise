imgG = imread("../img/noise/gaussian.png");
imgP = imread("../img/noise/poisson.png");
imgSP = imread("../img/noise/salt_pepper.png");

imwrite(bm3dFilter(imgG), "../img/bm3d/gaussian.png");
imwrite(bm3dFilter(imgP), "../img/bm3d/poisson.png");
imwrite(bm3dFilter(imgSP), "../img/bm3d/salt_pepper.png");


function [result2] = bm3dFilter(img)

    img = double(img);
    sigma = sqrt(0.005);
    blockWidth = 4;
    searchWidth = 19;
    selectionNum = 8;
    result1 = zeros(size(img));
    result2 = zeros(size(img));
    
    for channel = 1 : 3
        result1(:, :, channel) = first_step(img(:, :, channel), sigma, blockWidth, searchWidth, selectionNum);
        result2(:, :, channel) = second_step(img(:, :, channel), result1(:, :, channel), sigma, blockWidth, searchWidth, selectionNum);
    end
    result2 = uint8(result2);
end


function [result] = first_step(img, sigma, blockWidth, searchWidth, selectionNum)

    blockSize = 2 * blockWidth + 1;
    img = padarray(img, [searchWidth searchWidth], 'symmetric', 'both');
    [m, n] = size(img);
    numerator = zeros(m, n);
    denominator = zeros(m, n);
    
    for i = searchWidth + 1 : m - searchWidth
 
        for j = searchWidth + 1 : n - searchWidth
            
            window = img(i - searchWidth : i + searchWidth, j - searchWidth : j + searchWidth);
            blocksVectors = double(im2col(window, [blockSize blockSize], 'sliding'));

            centerBlockVectors = reshape(img(i - blockWidth : i + blockWidth, j - blockWidth : j + blockWidth), [], 1);
            dist = zeros(size(blocksVectors, 2), 1);
            for k = 1 : size(blocksVectors, 2)
                tmp = centerBlockVectors - blocksVectors(:, k);
                dist(k) = sum(tmp .^ 2);
            end

            [~, idxs] = sort(dist);
            idxs = idxs(1 : selectionNum);
            blocksVectors = blocksVectors(:, idxs);
            tmp = zeros(blockSize, blockSize, selectionNum);
            for k = 1 : selectionNum
                tmp(:, :, k) = reshape(blocksVectors(:, k), [blockSize blockSize]);
            end

            tmp = dct(tmp);
            tmp = wthresh(tmp, 'h', 5 * sigma);
            zeroNums = sum(tmp(:) > 0);
            if zeroNums > 0
                weight = 1 / zeroNums;
            else
                weight = 1;
            end

            tmp = idct(tmp);
            for k = 1 : selectionNum
                numerator(i - blockWidth: i + blockWidth, j - blockWidth: j + blockWidth) = numerator(i - blockWidth: i + blockWidth, j - blockWidth: j + blockWidth) + weight * tmp(:, :, k);
                denominator(i - blockWidth: i + blockWidth, j - blockWidth: j + blockWidth) = denominator(i - blockWidth: i + blockWidth, j - blockWidth: j + blockWidth) + weight;
            end
        end
    end
    result = numerator ./ denominator;
    result = result(searchWidth + 1 : end - searchWidth, searchWidth + 1 : end - searchWidth);
end


function [result] = second_step(noisyImg, prevRes, sigma, blockWidth, searchWidth, selectionNum)

    blockSize = 2 * blockWidth + 1;
    noisyImg = padarray(noisyImg, [searchWidth searchWidth], 'symmetric', 'both');
    [m, n] = size(noisyImg);
    prevRes = padarray(prevRes, [searchWidth searchWidth], 'symmetric', 'both');
    numerator = zeros(m, n);
    denominator = zeros(m, n);

    for i = searchWidth + 1 : m - searchWidth
        for j = searchWidth + 1 : n - searchWidth
            window1 = noisyImg(i - searchWidth : i + searchWidth, j - searchWidth : j + searchWidth);
            window2 = prevRes(i - searchWidth : i + searchWidth, j - searchWidth : j + searchWidth);
            blocksVectors1 = double(im2col(window1, [blockSize blockSize], 'sliding'));
            blocksVectors2 = double(im2col(window2, [blockSize blockSize], 'sliding'));
            centerBlockVectors = reshape(prevRes(i - blockWidth : i + blockWidth, j - blockWidth : j + blockWidth), [], 1);
            dist = zeros(size(blocksVectors1, 2), 1);
            for k = 1 : size(blocksVectors1, 2)
                tmp = centerBlockVectors - blocksVectors2(:, k);
                dist(k) = sum(tmp .^ 2);
            end
            [~, idxs] = sort(dist);
            idxs = idxs(1 : selectionNum);
            blocksVectors1 = blocksVectors1(:, idxs);
            blocksVectors2 = blocksVectors2(:, idxs);
            cubic1 = zeros(blockSize, blockSize, selectionNum);
            cubic2 = zeros(blockSize, blockSize, selectionNum);
            for k = 1 : selectionNum
                cubic1(:, :, k) = reshape(blocksVectors1(:, k), [blockSize blockSize]);
                cubic2(:, :, k) = reshape(blocksVectors2(:, k), [blockSize blockSize]);
            end
            cubic1 = dct(cubic1);
            cubic2 = dct(cubic2);
            weight = zeros(selectionNum, 1);
            for k = 1 : selectionNum
                tmp = norm(cubic2(:, :, k), 1) ^ 2;
                weight(k) = tmp / (tmp + sigma ^ 2);
            end
            for k = 1 : selectionNum
                cubic1(:, :, k) = cubic1(:, :, k) * weight(k);
            end
            cubic1 = idct(cubic1);
            weight = 1 / sum(weight(:) .^ 2);
            for k = 1 : selectionNum
                numerator(i - blockWidth: i + blockWidth, j - blockWidth: j + blockWidth) = numerator(i - blockWidth: i + blockWidth, j - blockWidth: j + blockWidth) + weight * cubic1(:, :, k);
                denominator(i - blockWidth: i + blockWidth, j - blockWidth: j + blockWidth) = denominator(i - blockWidth: i + blockWidth, j - blockWidth: j + blockWidth) + weight;
            end

        end

    end
    result = numerator ./ denominator;
    result = result(searchWidth + 1 : end - searchWidth, searchWidth + 1 : end - searchWidth);
    
end
