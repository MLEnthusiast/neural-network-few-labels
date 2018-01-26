function [tr_qb, tr_wv] = spatio_spectral_sample_reduction(bm_spectral, band_qb, band_wv, bm_band_texture)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    rows = size(band_qb, 1); % height
    cols = size(band_qb, 2); % width
    
    window_size = 8; % this number should be able to divide rows and cols
    block_pixel_density = zeros(rows / window_size, cols / window_size);
    block_size = window_size * window_size;
    
    for i = 1 : rows / window_size
        for j = 1 : cols / window_size
            i_img_start = (i - 1) * window_size + 1; % image row index start
            i_img_end = i * window_size; % image row index stop
            
            j_img_start = (j - 1) * window_size + 1; % image row index start
            j_img_end = j * window_size; % image row index stop
            
            temp_window = bm_band_texture(i_img_start:i_img_end, j_img_start:j_img_end);
            nbr_changes = sum(sum(temp_window > 0));
            block_pixel_density(i, j) = nbr_changes / block_size;
        end
    end
end

