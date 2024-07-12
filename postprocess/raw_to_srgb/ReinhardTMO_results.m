
srgb_folder = 'F:\HDR_results_2023data\Our_model\sRGB';
savefolder = 'F:\HDR_results_2023data\Our_model\Reinhard';


if ~exist(savefolder)
    mkdir(savefolder)
end

tifflist = dir(srgb_folder);
for i = 3: length(tifflist)
    tifffile = tifflist(i).name;
    tiffpath = fullfile(srgb_folder, tifffile);
    img = imread(tiffpath);
    img = im2double(img);
    imgTMO = ReinhardTMO(img);
    imgout = GammaTMO(imgTMO, 2.2,0,1);
    imgout = uint16(imgout*2^16);
    imwrite(imgout, fullfile(savefolder, tifffile))
end

    