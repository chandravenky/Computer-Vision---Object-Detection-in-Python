% Simple example of feature extraction and unsupervised learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load supplementary information
%load likes

% Set parameters
numimages = 307;
numcluster = 3;
numbins = 5;

% Initialize matrix to hold the data
color = zeros(numimages, 1);
brightness = zeros(numimages, numbins);
numlines = zeros(numimages, 1);
numcorners = zeros(numimages, 1);
hog = zeros(numimages, numbins);
lbp = zeros(numimages, 59);

for i = 1:numimages
  imgdata = imread(sprintf('img500_%d.jpg', i));
  
  % Colorfulness feature
  hsv = rgb2hsv(imgdata);  
  color(i) = mean(mean(hsv(:,:,1)));
  
  % Brightness feature
  gray = rgb2gray(imgdata);  
  [COUNTS,BINS] = imhist(gray,numbins);
  brightness(i,:) = COUNTS;
  
  % Number of lines
  BW = edge(gray,'canny');
  [H,theta,rho] = hough(BW);
  P = houghpeaks(H,5);
  lines = houghlines(BW,theta,rho,P);
  numlines(i) = size(lines,2);
  
  % Corner extraction
  C = corner(gray);
  numcorners(i) = length(C);
  
  % Histogram of oriented gradients (HOG)
  [featureVector,hogVisualization] = extractHOGFeatures(gray);
  [COUNTS,BINS] = imhist(featureVector,numbins);
  hog(i,:) = COUNTS;
  
  % Local binary patterns
  L = extractLBPFeatures(gray);
  lbp(i,:) = L;
end

% Create index for each of the images
index = [1:1:numimages]';

% Collect the set of features
features = [color, brightness, numlines, numcorners, hog, lbp];

% K-means clustering to group images together
cluster_kmeans = kmeans(features, numcluster);

% Clustering using Gaussian Mixture Model
gmfit = fitgmdist(features,numcluster, 'RegularizationValue', 0.1);
cluster_gmm = cluster(gmfit,features);

% Store the image data into matrix
results = [index, features, cluster_kmeans, cluster_gmm];
csvwrite('results_lbp.csv', results);

% Label the images with cluster in the filename
for i = 1:numimages
   imgdata = imread(sprintf('img500_%d.jpg', i)); 
   imwrite(imgdata,sprintf('cluster_kmeans_img%d_%d.jpg', i, cluster_kmeans(i)));
   imwrite(imgdata,sprintf('cluster_gmm_img%d_%d.jpg', i, cluster_gmm(i)));
end