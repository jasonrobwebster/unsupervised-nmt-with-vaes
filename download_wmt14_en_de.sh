mkdir -p data/parallel/wmt14/en_de
cd data/parallel/wmt14/en_de
wget https://ukuxhumana-language-data.s3.amazonaws.com/parallel/wmt14/en_de.tar.gz
tar xcvf en_de.tar.gz
rm en_de.tar.gz