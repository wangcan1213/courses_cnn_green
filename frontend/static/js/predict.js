$(function () {
    let img_width, img_height, box_width, box_height, img_wh_ratio, box_wh_ratio, new_width, new_height;
    [img_width, img_height] = [$('#raw_img').width(), $('#raw_img').height()];
    [box_width, box_height] = [$('#raw_img_box').width(), $('#raw_img_box').height()];
    [img_wh_ratio, box_wh_ratio] = [img_width / img_height, box_width / box_height];
    if (img_wh_ratio >= box_wh_ratio) {
        [new_width, new_height] = [box_width, box_width / img_wh_ratio];
    } else {
        [new_width, new_height] = [box_height * img_wh_ratio, box_height];
    }
    $('.img_result')
        .width(new_width)
        .height(new_height);
    if (box_height > new_height) {
        let delta_height = box_height - new_height;
        $('.img-label-text-up')
            .css('position', 'relative')
            .css('top', `${delta_height-25}px`);
    }
})