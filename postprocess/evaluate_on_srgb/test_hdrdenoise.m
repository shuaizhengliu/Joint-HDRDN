gt_srgb_folder = 'D:\HDR_results\GT\GT_srgb';
res_srgb_folder = 'F:\HDR_results\Models\sRGB'; % modify your path
num=28; % modify the image number
psnr_mu_all = zeros(num, 1);
psnr_l_all = zeros(num, 1);
ssim_l_all = zeros(num, 1);
ssim_mu_all = zeros(num, 1);
hdrvdp_all = zeros(num, 1);

tifflist = dir(res_srgb_folder);
for i = 3: length(tifflist)
    i
    tifffile = tifflist(i).name;
    res_tiffpath = fullfile(res_srgb_folder, tifffile);
    gt_tiffpath = fullfile(gt_srgb_folder, tifffile);
    res = imread(res_tiffpath);
    gt = imread(gt_tiffpath);
    res = im2double(res);
    gt = im2double(gt);
    res_mu = mulog_tonemap(res, 5000);
    gt_mu = mulog_tonemap(gt, 5000);
    psnr_l = psnr(res, gt);
    psnr_l_all(i-2) = psnr_l;
    psnr_mu = psnr(res_mu, gt_mu);
    psnr_mu_all(i-2) = psnr_mu;
    ssim_l = ssim(res, gt);
    ssim_l_all(i-2) = ssim_l;
    ssim_mu = ssim(res_mu, gt_mu);
    ssim_mu_all(i-2) = ssim_mu;
    
    ppd = hdrvdp_pix_per_deg(24, [size(gt,2) size(gt, 1)], 0.5);
    hdrvdp_res = hdrvdp(res, gt, 'sRGB-display', ppd);
    hdrvdp_all(i-2) = hdrvdp_res.Q;
end

average_psnr_l = mean(psnr_l_all);
average_psnr_mu = mean(psnr_mu_all);
average_ssim_l = mean(ssim_l_all);
average_ssim_mu = mean(ssim_mu_all);
average_hdr_vdp2 = mean(hdrvdp_all);

fprintf('Your model')
fprintf('[Average] psnr_mu: %.2f, psnr_l: %.2f, ssim_mu: %.4f, ssim_l: %.4f, HDR-VDP: %.2f\n',average_psnr_mu,average_psnr_l,average_ssim_mu,average_ssim_l,average_hdr_vdp2);
%fprintf('[Average] psnr_mu: %.2f, psnr_l: %.2f, ssim_mu: %.4f, ssim_l: %.4f, HDR-VDP: %.2f\n',average_psnr_mu,average_psnr_l,average_ssim_mu,average_ssim_l);
    
