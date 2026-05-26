export BMR_PAPER_STRICT="${BMR_PAPER_STRICT:-1}"
export BMR_PAPER_MLP="${BMR_PAPER_MLP:-1}"
export BMR_PATTERN_BACKBONE="${BMR_PATTERN_BACKBONE:-paper_inception_v3}"
CUDA_VISIBLE_DEVICES=0 python ./UAMFD.py -train_dataset weibo \
                                        -test_dataset weibo \
                                        -batch_size 16 \
                                        -epochs 50 \
                                        -val 0 \
                                        -is_sample_positive 1.0 \
                                        -duplicate_fake_times 0 \
                                        -network_arch UAMFD \
                                        -is_filter 0 \
                                        -not_on_12 1
# -checkpoint /groupshare/CIKM_ying_output//gossip/19_814_89.pkl
