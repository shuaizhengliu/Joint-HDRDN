#!/bin/bash
NAME=`basename "$0"`
    python3 ./test_release.py \
       --gpus                          1\
       --workers                         4\
       --checkpoint                    './checkpoint/Joint_HDRDN.pth'\
       --train_opt_file                './checkpoint/opts.pth'\
       --testfolder                       '/home/notebook/data/group/liushuaizheng/HDR_Denoise_Training/HDR_dynamic_data/Integral_test_p256_noborder' '/home/notebook/data/group/liushuaizheng/HDR_Denoise_Training/HDR_tranlate_static_data/Maxdisp20_randint/Test_data_p256_noborder'\
       --ori_testfolder                   '/home/notebook/data/group/liushuaizheng/HDR_Denoise_Training/HDR_dynamic_data/Integral_test' '/home/notebook/data/group/liushuaizheng/HDR_Denoise_Training/HDR_tranlate_static_data/Maxdisp20_randint/Test_data'\
       --model_module                        'Joint_HDRDN'\
       --model_type                          'Joint_HDRDN'\
       --patch_size                          256 \
       --result_dir                          './test_result'\

