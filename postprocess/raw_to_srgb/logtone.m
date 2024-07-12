function logtone_out = logtone(img, miu)
% the img should be the img 
logtone_out = log(1 + miu*img) / log(1+miu);

end