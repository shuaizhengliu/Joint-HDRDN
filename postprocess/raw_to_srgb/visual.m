load('H:\HDR_Dynamic\20220707\oppofindx3_able_dataset\2022-0707-1820-2918\denoised\iso-150-exp-0.01_meta.mat')

denoised_matfolder = 'D:\HDR_Dynamic_Dataset\test_result_0816\val_0816';
submats = dir(fullfile(denoised_matfolder, '*.mat'));
savefolder = 'D:\HDR_Dynamic_Dataset\test_result_0816\vis';
for i = 1:length(submats)
    matname = submats(i).name;
    matfile =fullfile(denoised_matfolder, matname);
    load(matfile);
    [out, ii] = run_pipeline(rawhdr, meta_data, 'normal', 'tone');
    savepath = fullfile(savefolder, [num2str(i), '.png']);
    imwrite(out, savepath)
end