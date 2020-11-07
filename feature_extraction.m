clc;
clear all;
close all;
I=imread('traffic.jpg');%to read the image 
imshow(I);%to show the image
% Igray=rgb2gray(I);
% imshow(Igray);
% level=0.63;
% ithresh=im2bw(Igray,level);
% imshowpair(I,ithresh,'montage');
%rgb color space
Im=double(I)/255;%to convert the image to double data type
Im=I;
rmat=Im(:,:,1);
gmat=Im(:,:,1);
bmat=Im(:,:,1);
% figure;
% subplot(2,2,1),imshow(rmat);
% title('red plane');
% subplot(2,2,2),imshow(gmat);
% title('green plane');
% subplot(2,2,3),imshow(bmat);
% title('blue plane');
% subplot(2,2,4),imshow(I);
% title('original');


levelr=0.63;
levelg=0.5;
levelb=0.4;
i1=im2bw(rmat,levelr);
i2=im2bw(gmat,levelg);
i3=im2bw(bmat,levelb);
Isum=(i1&i2&i3);
figure;
subplot(2,2,1),imshow(rmat);
title('red plane');
subplot(2,2,2),imshow(gmat);
title('green plane');
subplot(2,2,3),imshow(bmat);
title('blue plane');
subplot(2,2,4),imshow(Isum);
title('sum');



[~,threshold]=edge(Isum,'sobel');
fudgefactor=0.01;
TRs=edge(Isum,'sobel',threshold*fudgefactor);
figure;
imshow(TRs);



se90=strel('line',3,90);
se0=strel('line',3,0);
TRsdil=imdilate(TRs,[se90 se0]);
figure;
imshow(TRsdil);




Icomp=imcomplement(TRsdil);
figure;
imshow(Icomp);


Ifilled=imfill(TRsdil,'holes');
figure;
imshow(Ifilled);



% se90=strel('line',3,90);
se=strel('line',3,0);
% TRsdil=imdilate(Icomp,[se90 se0]);
% Ifilled= strel('ball',5,6);
Iopenned=imclose(TRsdil,se);
% figure;
imshow(Iopenned);




border2=imclearborder(Isum,8);
border=imclearborder(border2,4);
figure;
imshow(border);


noborder=-border+Iopenned;
no2border=noborder.*noborder;
figure;
imshow(noborder);
figure;
imshow(no2border);



% fill=imfill(noborder);
% figure;
% imshow(fill);

% 
% conn=bwconncomp(no2border,4);
% figure;
% imshow(conn);
% 
% 
% figure;
% clearborder=imclearborder(Iopenned,8);
% imshow(clearborder);


% subtr=-clearborder+Iopenned;
% figure;
% imshow(subtr);title('subtr');



[B,L] = bwboundaries(noborder,'noholes');
figure;
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on
for k = 1:length(B)
   boundary = B{k};
   plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
end


[B,L,N,A]=bwboundaries(L);

figure;  imshow(L);title('BackGround Detected');
figure;  imshow(L);title('Blob Detected');
L=logical(L);
hold on;
for k=1:length(L)

if(~sum(A(k,:)))
boundary = B{k}; 
plot(boundary(:,2), boundary(:,1), 'r','LineWidth',2);

for l=find(A(:,k))'
boundary = B{l};
plot(boundary(:,2), boundary(:,1), 'g','LineWidth',2);
end
end
end

blobMeasurements = regionprops(L,'all');
numberofcars = size(blobMeasurements,1)
s = regionprops(L,'centroid');



BlobAnalysis = vision.BlobAnalysis('MinimumBlobArea',100,'MaximumBlobArea',50000);


L=double(L);
 sd=imadjust(L);% adjust the image intensity values to the color map
 level=graythresh(sd);
 m=imnoise(sd,'gaussian',0,0.025);% apply Gaussian noise
 k=wiener2(m,[5,5]);%filtering using Weiner filter
 bw=imbinarize(k,level);
 bw2=imfill(bw,'holes');
 bw3 = bwareaopen(bw2,100);
 labeled = bwlabel(bw3,8);
 cc=bwconncomp(bw3);
 Densityoftraffic = cc.NumObjects/(size(bw3,1)*size(bw3,2));
 blobMeasurements = regionprops(labeled,'all');
 numberofcars = size(blobMeasurements,1);
 subplot(2,1,2) , imagesc(labeled), 
 se = strel('square', 9);
 sg=imdilate(labeled,se);
 figure();
 imagesc(sg)
title ('Detecting and counting of objects ');
 hold off;

 
 
 
 % L=logical(L);
%Display results% blob=BBox(L);
% figure;
% imshow(s);% [area,centroid,bbox] = step(BlobAnalysis,L);
% Ishape = insertShape(BlobAnalysis,'rectangle',bbox,'Color', 'green','Linewidth',6);

% imshow(BlobAnalysis);
% title('Detect');



% C = struct2table(connext); 
% A = table2array(C);
% % D=cell2int(A);
% imshow(A);



% Iregion=regionprops(Iopenned,'rectangle');
% [labeled,numObjects]=bwlabel(Iopenned,4);
% stats=regionprops(labeled,eccentricity,area,boundary);
% areas=[stats.area];
% eccentricties=[stats.Eccentricity];




% MV=imread(subtr);
% MV1=imread(subtr);
% A = double(rgb2gray(MV));%convert to gray
% B = double(rgb2gray(MV1));%convert 2nd image to gray
% [height, width] = size(A); %image size?
% h1 = figure(1);
 %Foreground Detection
%  thresh =10;
%  fr_diff = abs(A - B);
%  for j = 1:width
%  for k = 1:height
%  if (fr_diff(k,j)>thresh)
%  fg(k,j) = A(k,j);
%  else
%  fg(k,j) = 0;
%  end
%  end
%  end
%  subplot(2,1,1) , imagesc(MV), title (['Orignal Frame']);