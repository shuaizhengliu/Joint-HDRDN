function [output_CFA, iso, exp_time] = read_dng(dngpath)
% read single dng file
t = Tiff(dngpath, 'r');
output_CFA = read(t);
close(t);
meta_info = imfinfo(dngpath);

% crop to only valid pixels
x_origin = meta_info.ActiveArea(2)+1;
width = meta_info.DefaultCropSize(1);
y_origin = meta_info.ActiveArea(1)+1;
height = meta_info.DefaultCropSize(2);
output_CFA = double(output_CFA(y_origin:y_origin+height-1, x_origin:x_origin+width-1));
exp_time = meta_info.ExposureTime;
iso = meta_info.ISOSpeedRatings;
end

