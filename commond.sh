# check_feature
python3 check_feature.py images/A --feature orb --nFeatures 8000 --save --output features/orb8000
python3 check_feature.py images/T --feature orb --nFeatures 2000 --save --output features/orb2000

# 匹配照片
python3 check_match.py --kpfile1=features/orb8000/sll-1-orb.npz --kpfile2=features/orb8000/s11-1a-orb.npz --homography --save --output matches\A images/A/s11-1.jpg images/A/s11-1a.jpg
python3 check_match.py --kpfile1=features/orb2000/t20-3-orb.npz --kpfile2=features/orb8000/s11-1-orb.npz --homography --save --output matches\T images/T/t20-3.jpg images/A/s11-1.jpg
python3 check_match.py --kpfile1=features/orb2000/t20-4-orb.npz --kpfile2=features/orb8000/s11-1-orb.npz --homography --save --output matches\T images/T/t20-4.jpg images/A/s11-1.jpg
