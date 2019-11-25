for ((i = 1 ; i <= 70 ; i++)); do
  echo "creating folder: $i"
  mv ./${i}_1_*_*.jpg.chip.jpg ${i}/
done
