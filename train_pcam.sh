for percent in 0.02 0.005 0.001
do
    for alpha in 0.2 0.4 0.6 0.8 1.0
    do
    python Pcam/src/script/train_CustomCLIP.py --seed 1 --percent $percent --alpha $alpha
    done
done