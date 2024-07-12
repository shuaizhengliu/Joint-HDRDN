% matfolder = 'D:\HDR_results\LDRmat_all';
% metafolder = 'D:\HDR_results\Test_scene_meta';
% srgb_folder = 'D:\HDR_results\Test_LDRs\sRGB';
% mu_folder = 'D:\HDR_results\Test_LDRs\mu';

% matfolder = 'D:\HDR_withoutGT_results\LDRmat_all';
% metafolder = 'D:\HDR_withoutGT_results\Test_scene_meta';
% srgb_folder = 'D:\HDR_withoutGT_results\Test_LDRs\sRGB';
% mu_folder = 'D:\HDR_withoutGT_results\Test_LDRs\mu';



if ~exist(srgb_folder, 'dir')
    mkdir(srgb_folder)
end

if ~exist(mu_folder, 'dir')
    mkdir(mu_folder)
end

matfiles = dir(fullfile(matfolder));
for i = 3:length(matfiles)
    matfile = matfiles(i).name;
    matfile
    [p_folder, scene_name, ext] = fileparts(matfile);
    matfile_path = fullfile(matfolder, matfile);
    load(matfile_path);
    scene_name_p = strsplit(scene_name, '_');
    scene_name_p = scene_name_p{1};
    cur_metafolder = fullfile(metafolder, scene_name_p);
    metafile = dir(fullfile(cur_metafolder, '*.mat'));
    metafile_name = metafile(1).name;
    load(fullfile(cur_metafolder, metafile_name));
    srgb = run_pipeline( rawldr, meta_data, 'normal', 'srgb');
    srgb_uint16 = uint16(srgb*2^16);
    imwrite(srgb_uint16, fullfile(srgb_folder,[scene_name, '.tiff']));
    mu = run_pipeline(srgb, meta_data, 'srgb', 'tone');
    mu_uint16 = uint16(mu*2^16);
    imwrite(mu_uint16, fullfile(mu_folder,[scene_name, '.tiff']));
    
end