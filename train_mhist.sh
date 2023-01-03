for percent in 0.5 0.2 0.1 0.05 0.01
do
    for alpha in 0.2 0.4 0.6 0.8 1.0
    do
    python MHIST/src/script/train_CustomCLIP.py --seed 1 --percent $percent --alpha $alpha
    done
done