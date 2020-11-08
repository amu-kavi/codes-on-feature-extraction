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