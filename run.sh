# python cli_main.py --image ./KAIR/testsets/real3/palace.png --instruction "先灰度化再顺时针旋转90度" --output ./output/test_out.png
# python cli_main.py --image ./KAIR/testsets/real3/palace.png --instruction "去噪" --output ./output/palace_test_out.png
python cli_main.py --image ./KAIR/testsets/real3/palace.png --instruction "先去噪再超分" --output ./output/palace_test_out2.png
python cli_main.py --image ./KAIR/testsets/real3/palace.png --instruction "噪声少一点" --output ./output/palace_test_out3.png