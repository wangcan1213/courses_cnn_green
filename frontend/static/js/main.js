function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        var img = new Image();
        reader.onload = function (e) {
            img.src = e.target.result;
            img.onload = function () {
                let img_width, img_height, box_width, box_height, img_wh_ratio, box_wh_ratio, new_width, new_height;
                [img_width, img_height] = [this.width, this.height];
                [box_width, box_height] = [$('.imagebox').width(), $('.imagebox').height()];
                [img_wh_ratio, box_wh_ratio] = [img_width / img_height, box_width / box_height];
                if (img_wh_ratio >= box_wh_ratio) {
                    [new_width, new_height] = [box_width, box_width / img_wh_ratio];
                } else {
                    [new_width, new_height] = [box_height * img_wh_ratio, box_height];
                }
                $('#blah')
                    .attr('src', e.target.result)
                    .width(new_width)
                    .height(new_height);
                $('#preview_text').hide()
                $('#preview1').removeClass('preview_wait')
            }
        };

        reader.readAsDataURL(input.files[0]);
    }
}