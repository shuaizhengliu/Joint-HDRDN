function [ raw_data, meta_data ] = Load_Data_and_Metadata_from_DNG( image_name )
%Load_Data_and_Metadata_from_DNG Reads and returns raw image data and 
% metadata from DNG file "image_name"
% this version has crop the area

    t = Tiff(char(image_name), 'r');
    if t.getTag('BitsPerSample') ~= 16 % raw from DNG should be 16-bit
        try
            offsets = getTag(t, 'SubIFD');
            setSubDirectory(t, offsets(1));
        catch 
        end
    end
    raw_data = read(t);
    close(t);
    meta_data = imfinfo(char(image_name));
    
    %crop area
    cropArea = GetCropArea(meta_data); % zero-based index, I guess!
    raw_data = raw_data(cropArea(1)+1:cropArea(3)+0, ...
                                  cropArea(2)+1:cropArea(4)+0);
    
    
end

