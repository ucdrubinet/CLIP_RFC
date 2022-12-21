python Pcam/src/script/train_CustomCLIP.py --seed 1 --percent 0.05 --alpha 1.0
for percent in 0.01
do
    for alpha in 0 0.2 0.4 0.6 0.8 1.0
    do
    python Pcam/src/script/train_CustomCLIP.py --seed 1 --percent $percent --alpha $alpha
    done
done