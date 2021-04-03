Installing on existing repository

```bash
1041  2021-04-03 08:34:13  pip install tensorboard cmake
 1042  2021-04-03 08:34:35  pip install torch==1.8 torchvision==0.9 -f https://download.pytorch.org/whl/cu101/torch_stable.html
 1043  2021-04-03 08:37:20  pip install 'git+https://github.com/facebookresearch/fvcore'
 1044  2021-04-03 08:38:14  pip install 'git+https://github.com/facebookresearch/detectron2.git'
 1045  2021-04-03 08:39:07  gcc
 1046  2021-04-03 08:39:11  gcc --version
 1047  2021-04-03 08:39:19  pip list | grep cmake
 1048  2021-04-03 08:39:30  gcc -std=c++14
 1049  2021-04-03 08:40:32  ls
 1050  2021-04-03 08:41:49  cd /att/nobackup/jacaraba/
 1051  2021-04-03 08:41:49  ls
 1052  2021-04-03 08:41:50  git clone https://github.com/facebookresearch/detectron2
 1053  2021-04-03 08:42:16  pip install 'git+https://github.com/facebookresearch/detectron2.git'
 1054  2021-04-03 08:43:05  module avail
 1055  2021-04-03 08:43:17  module load gcc/8.4.0
 1056  2021-04-03 08:43:19  pip install 'git+https://github.com/facebookresearch/detectron2.git'
 ```
 
 Dockerfile for later is located in the repository tree.
 
 
 
