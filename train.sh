# ================ Edge Esmeralda Hackathon Project: Keoni ================
# This is an open source reproduction of ProGen2's small protein model. ProGen2
# small is a LLM-based model published in 2022, when it was a bleeding edge
# protein LLM model.
#  Paper link: https://arxiv.org/pdf/2206.13517
# 
# ProGen2 uses approximately the same parameters as gpt2. This means we can use
# Karpathy's llm.c - a clean, dependency-less CUDA implementation allowing you
# to train gpt-2 from scratch for about $20. Expanding on that work, I created
# a custom tokenizer for the ProGen2 data sources that is simple and
# easy-to-read, and am retraining the model from scratch.
#
# This is intended to be an educational resource for people wondering how
# LLMs can interoperate with biological data.

# Install go
GO_VERSION="1.22.4"
sudo apt update
sudo apt install -y wget tar # make sure tar is instaled
wget https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz # get current go version
sudo rm -rf /usr/local/go # clear previous installs
sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz # extract tarball to /usr/local
rm go${GO_VERSION}.linux-amd64.tar.gz # clear tarball
echo "export PATH=\$PATH:/usr/local/go/bin" >> ~/.profile # set environmental variables
source ~/.profile

# Clone dnadesign repo and build tokenizer
git clone https://github.com/Koeng101/dnadesign.git
cd dnadesign/lib/tokenizer/cli
go build -o tokenizer
mv tokenizer.go ~

# Pull uniprot uniref90. This is a massive file (44G), so we are going
# put it on an external drive (or block storage on a vm)
cd /mnt/ext/
wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
cd ~

# Now, let's tokenize the input. This takes a long time (~2-3 hours) and outputs
# a large number of sharded files (124G worth)
mkdir /mnt/ext/data
./tokenizer --outputDir /mnt/ext/data --unirefInput /mnt/ext/uniref90.fasta.gz

# ====
# We have the tokens, now continue llm.c
# ====

# install cudnn so we can use FlashAttention and run fast (optional)
# https://developer.nvidia.com/cudnn-downloads
# for me, CUDA 12 (run `nvcc --version`) running on Linux x86_64 Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcudnn9-dev-cuda-12

# "install" cudnn-frontend to ~/
git clone https://github.com/NVIDIA/cudnn-frontend.git

# install MPI (optional, if you intend to use multiple GPUs)
sudo apt install openmpi-bin openmpi-doc libopenmpi-dev

# tokenize the FineWeb dataset 10B tokens sample (takes ~1 hour, get lunch?)
# writes ~19GB of raw GPT-2 tokens to dev/data/fineweb10B
# and ~46GB in ~/.cache/huggingface/datasets/HuggingFaceFW___fineweb
git clone https://github.com/karpathy/llm.c.git
cd llm.c

# ====
# Before we run our training, we need to apply some changes, mostly to the EOT
# and vocab size parameters
# ====
sed -i 's/50257/30/' "train_gpt2.cu" # change vocab_size first instance
sed -i 's/50257/30/' "train_gpt2.cu" # change vocab_size second instance
sed -i 's/eot_token = tokenizer.eot_token/eot_token = 0/' "train_gpt2.cu"

# compile llm.c (mixed precision, with cuDNN flash-attention)
# first compilation is ~1 minute, mostly due to cuDNN
make train_gpt2cu USE_CUDNN=1

# train on a single GPU
./train_gpt2cu \
    -i "/mnt/ext/data/*train*.bin" \
    -j "/mnt/ext/data/*val*.bin" \
    -o log124M \
    -e "d12" \
    -b 32 -t 1024 \
    -d 524288 \
    -r 1 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.0 \
    -u 700 \
    -n 5000 \
    -v 250 -s 20000 \
    -h 1
# batch size changed to 32 for running on a single 24gb VRAM graphics card

# if you have multiple GPUs (e.g. 8), simply prepend the mpi command, e.g.:
# mpirun -np 8 ./train_gpt2cu \ ... (the rest of the args are same)

# ====
# Training takes ~261hr on 1 A10 GPU
# ====
