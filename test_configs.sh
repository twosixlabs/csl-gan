for dataset in MNIST CelebA
do
  for privacy_method in gc is
  do
    echo ==== $dataset $privacy_method ====
    echo [ Unconditional ]
    timeout 60s python3 train.py $dataset -tss 1000 -dpm $privacy_method -nms 1 --mean_sample_size 10
    echo [ Conditional ]
    timeout 60s python3 train.py $dataset -tss 1000 -dpm $privacy_method -nms 1 --mean_sample_size 10 --conditional
  done
done
